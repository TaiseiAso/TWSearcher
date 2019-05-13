# coding: utf-8

"""ツイート内容に関連するウィキペディアの文をTF-IDF+Cos類似度で検索する"""
__author__ = "Aso Taisei"
__version__ = "1.0.0"
__date__ = "13 May 2019"


# 必要なモジュールをインポート
import os
import yaml
import numpy as np
import math


class Searcher():
    """TF-IDF+Cos類似度検索するクラス"""
    def __init__(self, config):
        """
        コンストラクタ
        @param config 設定ファイルの情報
        """
        # 保存ファイル名の情報取得
        fn = config['filename']
        prt_fn = fn['part_file']
        self.twi_fp = "data/" + fn['twitter_file'] + ".txt"
        self.wik_fp = "data/" + fn['wikipedida_file'] + ".txt"
        self.twi_prt_fp = "data/" + fn['twitter_file'] + "_" + prt_fn + ".txt"
        self.wik_prt_fp = "data/" + fn['wikipedida_file'] + "_" + prt_fn + ".txt"
        self.rel_fp = "dump/" + fn['relation_file'] + ".txt"
        self.wik_cut_fp = "dump/" + fn['wikipedida_file'] + ".txt"
        self.wik_prt_cut_fp = "dump/" + fn['wikipedida_file'] + "_" + prt_fn + ".txt"

        # 検索数
        self.dump = config['dump']

        # 関連しない文を除去したウィキペディアコーパスを保存するかどうか
        # 対応関係もそのコーパスへの番号となる
        self.cut = config['cut']

        # 品詞を検索に使用するかどうか
        use = config['part']['use']
        self.use_list = [
            use['noun_main'], use['verb_main'], use['adjective_main'],
            use['noun_sub'], use['verb_sub'], use['adjective_sub'],
            use['adverb'], use['particle'], use['auxiliary_verb'],
            use['conjunction'], use['prefix'], use['filler'],
            use['impression_verb'], use['three_dots'], use['phrase_point'],
            use['reading_point'], use['other']
        ]

        # 品詞のトークンを取得
        tk = config['part']['token']
        self.token_list = [
            tk['noun_main'], tk['verb_main'], tk['adjective_main'],
            tk['noun_sub'], tk['verb_sub'], tk['adjective_sub'],
            tk['adverb'], tk['particle'], tk['auxiliary_verb'],
            tk['conjunction'], tk['prefix'], tk['filler'],
            tk['impression_verb'], tk['three_dots'], tk['phrase_point'],
            tk['reading_point'], tk['other']
        ]

    def del_part(self, text, part):
        """
        指定した品詞のみを除外する
        @param text 空白で分かち書きされたテキスト
        @param part 品詞列
        @return 品詞除去された単語列
        """
        result = []
        words, parts = text.strip().split(), part.strip().split()
        for word, part in zip(words, parts):
            if part in self.token_list:
                part_idx = self.token_list.index(part)
                if self.use_list[part_idx]:
                    result.append(word)
        return result

    def search(self):
        """ツイート内容に関連するウィキペディアの文の対応関係をファイルに保存する"""
        if not os.path.isdir("data"):
            print("no data folder")
            return

        if not os.path.isfile(self.twi_fp):
            print("no " + self.twi_fp + " file")
            return

        if not os.path.isfile(self.wik_fp):
            print("no " + self.wik_fp + " file")
            return

        # dumpフォルダがなければ作成する
        if not os.path.isdir("dump"):
            os.mkdir("dump")

        # dumpフォルダ内のファイルをすべて削除
        for root, _, files in os.walk("dump", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))

        tf_lsts = []
        idf = {}
        all = 0
        for fp, prt_fp in [[self.twi_fp, self.twi_prt_fp], [self.wik_fp, self.wik_prt_fp]]:
            tf_lst = []

            f = open(fp, 'r', encoding='utf-8')
            if os.path.isfile(prt_fp):
                prt = True
                f_prt = open(prt_fp, 'r', encoding='utf-8')
            else:
                prt = False

            line = f.readline()
            line_prt = f_prt.readline() if prt else None

            while line:
                all += 1

                if prt:
                    words = self.del_part(line, line_prt)
                else:
                    words = line.strip().split()

                tf, ex_lst = {}, []
                for word in words:
                    if word not in ex_lst:
                        tf[word] = 1
                        ex_lst.append(word)
                    else:
                        tf[word] += 1

                len_words = len(words)
                tf_lst.append({k: v / len_words for (k, v) in tf.items()})

                for ex in ex_lst:
                    if ex in idf:
                        idf[ex] += 1
                    else:
                        idf[ex] = 1

                line = f.readline()
                line_prt = f_prt.readline() if prt else None

            f.close()
            if prt:
                f_prt.close()

            tf_lsts.append(tf_lst)

        twi_tf_idf_lst = [{k: v * math.log(all / idf[k]) for (k, v) in tf.items()} for tf in tf_lsts[0]]
        wik_tf_idf_lst = [{k: v * math.log(all / idf[k]) for (k, v) in tf.items()} for tf in tf_lsts[1]]

        twi_norm_lst = [np.linalg.norm(list(twi_tf_idf.values()), ord=2) for twi_tf_idf in twi_tf_idf_lst]
        wik_norm_lst = [np.linalg.norm(list(wik_tf_idf.values()), ord=2) for wik_tf_idf in wik_tf_idf_lst]

        sim_rank_lst = []
        for twi_tf_idf, twi_norm in zip(twi_tf_idf_lst, twi_norm_lst):
            sim_lst = []
            for wik_tf_idf, wik_norm in zip(wik_tf_idf_lst, wik_norm_lst):
                dot = 0
                for twi_key in twi_tf_idf.keys():
                    if twi_key in wik_tf_idf:
                        dot += twi_tf_idf[twi_key] * wik_tf_idf[twi_key]
                sim_lst.append(dot / (twi_norm * wik_norm))
            sim_rank_lst.append(np.argsort(sim_lst)[::-1][:self.dump])

        if self.cut:
            rel_lst = []
            for sim_rank in sim_rank_lst:
                for idx in sim_rank:
                    rel_lst.append(idx)
            rel_lst = list(set(rel_lst))

            f = open(self.wik_fp, 'r', encoding='utf-8')
            f_cut = open(self.wik_cut_fp, 'w', encoding='utf-8')
            if os.path.isfile(self.wik_prt_fp):
                prt = True
                f_prt = open(self.wik_prt_fp, 'r', encoding='utf-8')
                f_prt_cut = open(self.wik_prt_cut_fp, 'w', encoding='utf-8')
            else:
                prt = False

            idx = 0
            for rel in rel_lst:
                while idx <= rel:
                    line = f.readline()
                    line_prt = f_prt.readline() if prt else None
                    idx += 1
                f_cut.write(line)
                if prt:
                    f_prt_cut.write(line_prt)

            f.close()
            f_cut.close()
            if prt:
                f_prt.close()
                f_prt_cut.close()

            for i in range(len(sim_rank_lst)):
                for j in range(len(sim_rank_lst[i])):
                    sim_rank_lst[i][j] = rel_lst.index(sim_rank_lst[i][j])

        with open(self.rel_fp, 'w', encoding='utf-8') as f:
            for sim_rank in sim_rank_lst:
                add = ""
                for sr in sim_rank:
                    add += str(sr) + " "
                f.write(add.strip() + "\n")


def similar_search(config):
    """
    ツイート内容に関連するウィキペディアの文をTF-IDF+Cos類似度で検索
    @param config 設定ファイルの情報
    """
    searcher = Searcher(config)
    searcher.search()


if __name__ == '__main__':
    # 設定ファイルを読み込む
    config = yaml.load(stream=open("config/config.yml", 'rt', encoding='utf-8'), Loader=yaml.SafeLoader)

    # TF-IDF+Cos類似度で検索
    similar_search(config)
