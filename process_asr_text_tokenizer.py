
import argparse
import json
import logging
import os
from typing import List, Optional

import tokenizers

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.utils.data_utils import DataStoreObject

parser = argparse.ArgumentParser(description='Create tokenizer')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--manifest", default=None, type=str, help='Comma separated list of manifest files')
group.add_argument("--data_file", default=None, help='data file from which to create tokenizer model')
parser.add_argument("--data_root", required=True, default=None, type=str, help='Output directory')
parser.add_argument("--vocab_size", default=1024, type=int, help='Vocabulary size')
parser.add_argument("--tokenizer", default="wpe", choices=["spe", "wpe"], help='Type of tokenization to perform')
parser.add_argument(
    "--spe_type",
    default="bpe",
    choices=['bpe', 'unigram', 'char', 'word'],
    help='Type of the SentencePiece model. Can be `bpe`, `unigram`, `char` or `word`.'
    'Used only if --tokenizer == `spe`',
)
parser.add_argument(
    '--spe_character_coverage',
    type=float,
    default=1.0,
    help="Character coverage percentage for SentencePiece tokenization. For languages "
    "with large vocabulary, should be close to 0.9995, otherwise kept as 1.0",
)
parser.add_argument('--spe_bos', action='store_true', help='Add <s> token to SentencePiece Tokenizer.')
parser.add_argument('--spe_eos', action='store_true', help='Add </s> token to SentencePiece Tokenizer.')
parser.add_argument('--spe_pad', action='store_true', help='Add <pad> token to SentencePiece Tokenizer.')
parser.add_argument(
    '--spe_user_defined_symbols', default=None, type=str, nargs='+', help='User defined symbols for SentencePiece'
)
parser.add_argument(
    '--spe_control_symbols', default=None, type=str, nargs='+', help='Control symbols for SentencePiece'
)
parser.add_argument('--spe_split_digits', action='store_true', help='Split digits into separate tokens.')

parser.add_argument(
    '--spe_sample_size',
    type=int,
    default=-1,
    help="Samples the dataset by `sample_size` if positive integer, otherwise uses whole dataset",
)
parser.add_argument('--spe_train_extremely_large_corpus', action='store_true', help='')
parser.add_argument(
    '--spe_max_sentencepiece_length',
    type=int,
    default=-1,
    help='Limit the maximum number of tokens in each SentencePiece subword. '
    'Must be a positive integer > 0. By default places no limit on subword length.',
)
parser.add_argument(
    '--spe_no_split_by_unicode_script',
    dest='spe_split_by_unicode_script',
    action='store_false',
    help="Don't use Unicode script to split sentence pieces.",
)
parser.add_argument(
    '--spe_byte_fallback',
    dest='spe_byte_fallback',
    action='store_true',
    help="If <unk>, fallback to a byte sequence of the characters.",
)
parser.add_argument('--no_lower_case', dest='lower_case', action='store_false')
parser.add_argument("--log", action='store_true')
parser.set_defaults(log=False, lower_case=True, spe_train_extremely_large_corpus=False)
args = parser.parse_args()


def __build_document_from_manifests(
    data_root: str, manifests: str,
):
    if ',' in manifests:
        manifests = manifests.split(',')
    else:
        manifests = [manifests]

    document_dir = os.path.join(data_root, 'text_corpus')
    if not os.path.exists(document_dir):
        os.makedirs(document_dir)

    document_path = os.path.join(document_dir, 'document.txt')

    if os.path.exists(document_path):
        logging.info('Corpus already exists at path : %s', document_path)
        return document_path

    num_lines = 0
    with open(document_path, 'w') as out_writer:
        for manifest in manifests:
            with open(DataStoreObject(manifest).get(), 'r') as in_reader:
                for line in in_reader:
                    item = json.loads(line)
                    text = item['text']

                    out_writer.write(text + '\n')
                    out_writer.flush()

                    num_lines += 1

            logging.info(f"Finished extracting manifest : {manifest}")

        logging.info("Finished extracting all manifests ! Number of sentences : {}".format(num_lines))
    return document_path


def __process_data(
    text_path: str,
    dst_folder: str,
    vocab_size: int,
    tokenizer_type: str,
    spe_type: str,
    spe_character_coverage: float,
    spe_train_extremely_large_corpus: bool,
    spe_sample_size: int,
    spe_max_sentencepiece_length: int,
    spe_split_by_unicode_script: bool,
    spe_bos: bool,
    spe_eos: bool,
    spe_pad: bool,
    spe_control_symbols: Optional[List[str]],
    spe_user_defined_symbols: Optional[List[str]],
    spe_byte_fallback: bool,
    spe_split_digits: bool,
    lower_case: bool,
):
    
    if tokenizer_type == 'spe':

        # Prepare directory of tokenizer
        if spe_max_sentencepiece_length > 0:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}_max_{}').format(
                tokenizer_type, spe_type, vocab_size, spe_max_sentencepiece_length
            )
        else:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}').format(
                tokenizer_type, spe_type, vocab_size
            )

        if spe_pad:
            tokenizer_dir = f'{tokenizer_dir}_pad'
        if spe_bos:
            tokenizer_dir = f'{tokenizer_dir}_bos'
        if spe_eos:
            tokenizer_dir = f'{tokenizer_dir}_eos'

        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        if os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.model')):
            logging.warning("Model file already exists, overriding old model file !")
            os.remove(os.path.join(tokenizer_dir, 'tokenizer.model'))

        # Build tokenizer
        tokenizer_path, vocab_path = create_spt_model(
            data_file=text_path,
            vocab_size=vocab_size,
            sample_size=spe_sample_size,
            do_lower_case=lower_case,
            output_dir=tokenizer_dir,
            tokenizer_type=spe_type,
            character_coverage=spe_character_coverage,
            train_extremely_large_corpus=spe_train_extremely_large_corpus,
            max_sentencepiece_length=spe_max_sentencepiece_length,
            split_by_unicode_script=spe_split_by_unicode_script,
            bos=spe_bos,
            eos=spe_eos,
            pad=spe_pad,
            control_symbols=spe_control_symbols,
            user_defined_symbols=spe_user_defined_symbols,
            byte_fallback=spe_byte_fallback,
            split_digits=spe_split_digits,
        )

    else:
        tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_v{}').format(tokenizer_type, vocab_size)

        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)

        tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=lower_case)

        tokenizer.train(text_path, vocab_size=vocab_size)
        tokenizer.save_model(tokenizer_dir)

    return tokenizer_dir


def main():
    data_root = args.data_root
    manifests = args.manifest
    data_file = args.data_file
    vocab_size = args.vocab_size
    tokenizer = args.tokenizer
    spe_type = args.spe_type
    spe_character_coverage = args.spe_character_coverage
    spe_sample_size = args.spe_sample_size
    spe_train_extremely_large_corpus = args.spe_train_extremely_large_corpus
    spe_max_sentencepiece_length = args.spe_max_sentencepiece_length
    spe_split_by_unicode_script = args.spe_split_by_unicode_script
    spe_bos, spe_eos, spe_pad = args.spe_bos, args.spe_eos, args.spe_pad
    spe_control_symbols = args.spe_control_symbols
    spe_user_defined_symbols = args.spe_user_defined_symbols
    spe_byte_fallback = args.spe_byte_fallback
    spe_split_digits = args.spe_split_digits
    lower_case = args.lower_case

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if args.log:
        logging.basicConfig(level=logging.INFO)

    if manifests:
        text_corpus_path = __build_document_from_manifests(data_root, manifests)
    else:
        text_corpus_path = data_file
    tokenizer_path = __process_data(
        text_corpus_path,
        data_root,
        vocab_size,
        tokenizer,
        spe_type,
        lower_case=lower_case,
        spe_character_coverage=spe_character_coverage,
        spe_sample_size=spe_sample_size,
        spe_train_extremely_large_corpus=spe_train_extremely_large_corpus,
        spe_max_sentencepiece_length=spe_max_sentencepiece_length,
        spe_split_by_unicode_script=spe_split_by_unicode_script,
        spe_bos=spe_bos,
        spe_eos=spe_eos,
        spe_pad=spe_pad,
        spe_control_symbols=spe_control_symbols,
        spe_user_defined_symbols=spe_user_defined_symbols,
        spe_byte_fallback=spe_byte_fallback,
        spe_split_digits=spe_split_digits,
    )

    print("Serialized tokenizer at location :", tokenizer_path)
    logging.info('Done!')


if __name__ == "__main__":
    main()