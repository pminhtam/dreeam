from span_marker import SpanMarkerModel
from transformers import AutoTokenizer
import torch


def preprocess_entity_one_sentence(sentence, entity_model, sent_id= 0):
    words = sentence.strip("").strip("").split(' ')
    entities = entity_model.predict(sentence)
    vertexSet_ = []
    for entity in entities:
        vertex_ = {}
        vertex_['name'] = entity['span']
        vertex_['sent_id'] = sent_id
        vertex_['type'] = entity['label']
        name = entity['span'].split(' ')
        # loc = sentence.find(name)
        try:
            loc_start = words.index(name[0])
            loc_end = loc_start + len(name)
            vertex_['pos'] = [loc_start, loc_end]
        except ValueError:
            continue
        vertexSet_.append(vertex_)
    return vertexSet_


def preprocess_entity(paragraph, entity_model, tokenizer):
    sentences = paragraph.split(" . ")
    vertexSet = []
    sent_id = 0
    sents = []
    for sent in sentences:
        # print(sent)
        vertexSet_ = preprocess_entity_one_sentence(sent, entity_model, sent_id)
        if len(vertexSet_) > 0:
            vertexSet.append(vertexSet_)
        sent_id += 1
        sents.append(tokenizer.tokenize(sent))
    data = {}
    data['vertexSet'] = vertexSet
    data['sents'] = sents
    data['title'] = "News"
    return data

from prepro import add_entity_markers, docred_rel2id
def preprocess_feature(data, tokenizer):
    pos_samples = 0
    neg_samples = 0
    max_seq_length=1024
    sample = data
    # print(sample)
    entities = sample['vertexSet']
    entity_start, entity_end = [], []
    # record entities
    for entity in entities:
        for mention in entity:
            sent_id = mention["sent_id"]
            pos = mention["pos"]
            entity_start.append((sent_id, pos[0],))
            entity_end.append((sent_id, pos[1] - 1,))

    # add entity markers
    sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)
    # print(sample)
    # print(entity_pos)
    # exit()
    # training triples with positive examples (entity pairs with labels)
    train_triple = {}

    if "labels" in sample:
        for label in sample['labels']:
            evidence = label['evidence']
            r = int(docred_rel2id[label['r']])

            # update training triples
            if (label['h'], label['t']) not in train_triple:
                train_triple[(label['h'], label['t'])] = [
                    {'relation': r, 'evidence': evidence}]
            else:
                train_triple[(label['h'], label['t'])].append(
                    {'relation': r, 'evidence': evidence})

    # entity start, end position
    entity_pos = []

    for e in entities:
        entity_pos.append([])
        assert len(e) != 0
        for m in e:
            start = sent_map[m["sent_id"]][m["pos"][0]]
            end = sent_map[m["sent_id"]][m["pos"][1]]
            label = m["type"]
            entity_pos[-1].append((start, end,))

    relations, hts, sent_labels = [], [], []

    for h, t in train_triple.keys():  # for every entity pair with gold relation
        relation = [0] * len(docred_rel2id)
        sent_evi = [0] * len(sent_pos)

        for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
            relation[mention["relation"]] = 1
            for i in mention["evidence"]:
                sent_evi[i] += 1

        relations.append(relation)
        hts.append([h, t])
        sent_labels.append(sent_evi)
        pos_samples += 1

    for h in range(len(entities)):
        for t in range(len(entities)):
            # all entity pairs that do not have relation are treated as negative samples
            if h != t and [h, t] not in hts:  # and [t, h] not in hts:
                relation = [1] + [0] * (len(docred_rel2id) - 1)
                sent_evi = [0] * len(sent_pos)
                relations.append(relation)

                hts.append([h, t])
                sent_labels.append(sent_evi)
                neg_samples += 1

    assert len(relations) == len(entities) * (len(entities) - 1)
    assert len(sents) < max_seq_length
    sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    feature = [{'input_ids': input_ids,
                'entity_pos': entity_pos,
                'labels': relations,
                'hts': hts,
                'sent_pos': sent_pos,
                'sent_labels': sent_labels,
                'title': sample['title'],
                }]

    return feature

if __name__ == '__main__':

    # Download from the 🤗 Hub
    entity_model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
    # Run inference
    sentence = "Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris"
    entities = entity_model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
    words = sentence.split(' ')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # tokens_wordpiece = tokenizer.tokenize(token)
    # print(entities)
    # print(preprocess_entity_one_sentence(sentence, entity_model))

    paragraph = "Roman Bernard Atwood -LRB- born May 28 , 1983 -RRB- is an American YouTube personality , comedian , vlogger and pranker . He is best known for his vlogs , where he posts updates about his life on a daily basis . His vlogging channel , `` RomanAtwoodVlogs '' , has a total of 3.3 billion views and 11.9 million subscribers . He also has another YouTube channel called `` RomanAtwood '' , where he posts pranks . His prank videos have gained over 1.4 billion views and 10.3 million subscribers . Both of these channels are in the top 100 most subscribed on YouTube , and he became the second YouTuber after Germ\u00e1n Garmendia to receive two Diamond Play Buttons for his two channels . "
    data = preprocess_entity(paragraph, entity_model, tokenizer)
    feat = preprocess_feature(data, tokenizer)


    from transformers import AutoConfig, AutoModel
    from model import DocREModel

    # print(feat)
    model_name_or_path = "bert-base-cased"
    num_class = 97
    num_labels = 4
    max_sent_num = 25
    evi_thresh = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_class,
    )
    config.transformer_type = "bert"
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = AutoModel.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    model = DocREModel(config, model, tokenizer,
                    num_labels=num_labels,
                    max_sent_num=max_sent_num,
                    evi_thresh=evi_thresh)
    model.to(device)
    model_path = "dreeam_models/bert_teacher_best.ckpt"
    model.load_state_dict(torch.load(model_path), strict=False)

    from run import load_input
    from utils import collate_fn
    import numpy as np
    feat_tensor = collate_fn(feat)
    inputs = load_input(feat_tensor,device)
    # print(inputs)
    outputs = model(**inputs)
    # print(outputs)
    preds, evi_preds = [], []
    scores, topks = [], []
    attns = []

    pred = outputs["rel_pred"]
    pred = pred.cpu().numpy()
    pred[np.isnan(pred)] = 0
    preds.append(pred)
    # print(pred)
    # print(outputs)
    if "scores" in outputs:
        scores.append(outputs["scores"].detach().cpu().numpy())
        topks.append(outputs["topks"].detach().cpu().numpy())

    if "evi_pred" in outputs:  # relation extraction and evidence extraction
        evi_pred = outputs["evi_pred"]
        evi_pred = evi_pred.detach().cpu().numpy()
        evi_preds.append(evi_pred)

    preds = np.concatenate(preds, axis=0)

    if scores != []:
        scores = np.concatenate(scores, axis=0)
        topks = np.concatenate(topks, axis=0)

    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)
    from evaluation import to_official
    official_results, results = to_official(preds, feat, evi_preds=evi_preds, scores=scores, topks=topks)

    # print(official_results)
    print(results)
    for re in results:
        pass