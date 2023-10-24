from tensorflow.python.keras.models import load_model
from main.data_helpers import load_data_x
import numpy as np
import tensorflow as tf
import os
import re
from utils.utils import log


class AsmModule:

    #? model config
    _model = None
    _model_path = ''
    _vocab_path = None
    _sequence_length = None


    def __init__(self, config):
        if config is None or 'model_path' not in config:
            log('[!][AsmModule] no `model_path` defined', 'warning')
            return
        self.change_config(config)

        self.clean_pattern = re.compile(r'(.*)(\||\│|\╎|\└|\\|\<|\>)\s(.*)')
        self.replace_var_pattern = re.compile(r'((0x|fcn\.|arg\_|var\_)([0-9a-z]{1,8}))')
        
        self._vocab_path = config['vocab_path']
        self._sequence_length = config['sequence_length']
        
        """ Load model """
        if os.path.isfile(self._model_path):
            self.session = tf.compat.v1.Session()
            self.graph = tf.compat.v1.get_default_graph()
            with self.graph.as_default():
                with self.session.as_default():
                    print('[ ][Asm_Module] tf graph initialised')
                    self._model = load_model(self._model_path)
        else: #? model_path not exist
            log('[!][AsmModule] `model_path` not exist', 'warning')

        return
    
    
    def change_config(self, config):
        if config is None:
            return

        #? if model_path is passed in config, load new model
        if 'model_path' in config and config['model_path'] != self._model_path:
            self._model_path = config['model_path']

            if os.path.isfile(self._model_path):
                self.session = tf.compat.v1.Session()
                self.graph = tf.compat.v1.get_default_graph()
                with self.graph.as_default():
                    with self.session.as_default():
                        print('[ ][Asm_Module][change_config] tf graph initialised')
                        self._model = load_model(self._model_path)
            else: #? model_path not exist
                log('[!][AsmModule][change_config] `model_path` not exist', 'warning')
                self._model = load_model(self._model_path)

        return



    def from_files(self, _map_ohash_inputs, callback):
        seq_datas = []
        for ohash,filepaths in _map_ohash_inputs.items():
            #? filepaths is an array of 3 paths: 
            # [
            #   '/home/mta-smad/smad-3/data/modules/prp-disasm/output/asm_cleaned/admin__2023-08-27_14-24-11__Lab_05-1.malware.asm', 
            #   '/home/mta-smad/smad-3/data/modules/prp-disasm/output/cfg/admin__2023-08-27_14-24-11__Lab_05-1.malware.jpg', 
            #   '/home/mta-smad/smad-3/data/modules/prp-disasm/output/cfg/admin__2023-08-27_14-24-11__Lab_05-1.malware.dot'
            # ]
            filepath = filepaths[0] #? asm path is the 1st path
            asm = [line.strip() for line in open(filepath, 'r').readlines()]
            seq_datas.append(asm)

        if self._model is None:
            log('[!][Asm_Module][change_config] `model` not found', 'error')
            #? return empty result for each item
            result = {ohash: '' for ohash in _map_ohash_inputs.keys()}
            callback(result)
            return


        """ Infer """
        X = load_data_x(seq_datas, sequence_length=self._sequence_length,
                              vocabulary_inv_path=self._vocab_path)

        with self.graph.as_default():
            with self.session.as_default():
                preds = [pred[0] for pred in self._model.predict(X)]
                lbl_preds = np.array([1 if pred > 0.99 else 0 for pred in preds]) #? only 1 or 0 (boolean result)

        print('[+][Asm_Module][from_files] lbl_preds, preds', lbl_preds, preds)

        #? Callbacks on finish
        result = {}
        note = {}
        k = 0
        for ohash in _map_ohash_inputs.keys():
            result[ohash] = bool(int(lbl_preds[k]))
            note[ohash] = float(preds[k])
            k += 1

        #! Call __onFinishInfer__ when the analysis is done. This can be called from anywhere in your code. In case you need synchronous processing
        callback(result, note)

        # return lbl_preds, preds

