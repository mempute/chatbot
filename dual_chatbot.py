


import numpy as np
import mempute as mp
from data import DatasetManager
import time


class KModel:
    def __init__(self, xsz=16, latent_sz=128, embede_dim = 128, nb=32, ep = None, name = None):

        self.dm = DatasetManager('chatbot') #option
        self.dm.load_vocab()

        self.seq_len = xsz * 2
        self.half_len = int(self.seq_len / 2) - 1

        ubase = 0 #option
        self.train_data_iter = self.dm.data_generator(nb, self.half_len, data_type='train', epoch=ep, under_line=ubase)
        self.test_data_iter = self.dm.data_generator(nb, self.half_len, data_type='test', epoch=ep, under_line=ubase)

        self.train_batch_iter = self.dm.data_generator(nb, self.half_len, data_type='train', epoch=ep, under_line=ubase)
        self.test_batch_iter = self.dm.data_generator(nb, self.half_len, data_type='test', epoch=ep, under_line=ubase)

        self.batch_size = nb
        self.logfp = None

        if name is None:
            self.trc = mp.tracer()
        else:
            self.trc = mp.tracer(0, name)
            self.model_name = name + '.log'
        #mp.modeset(self.trc, 1)
        #mp.lapset(self.trc, 1)
        #mp.setgpudev(self.trc, 1)
        #mp.npset(self.trc, 100)

        self.in_plus = 0

        if self.in_plus: self.iaccu_pred = self.half_len +1 #목표값 부분만 정확도 계산
        else: self.iaccu_pred = 0

        self.by = mp.flux(self.trc, [nb, self.seq_len, 1], mp.variable, mp.tfloat)
        if self.in_plus:
            self.ytar = mp.flux(self.trc, [nb, self.seq_len, 1], mp.variable, mp.tfloat)
        else:
            self.ytar = mp.flux(self.trc, [nb, self.half_len +1, 1], mp.variable, mp.tfloat)
        self.in_d = mp.flux(self.trc, [nb, self.half_len, 1], mp.variable, mp.tfloat)
        self.tar_d = mp.flux(self.trc, [nb, self.half_len, 1], mp.variable, mp.tfloat)
        self.zpad = mp.flux(self.trc, [nb, 1, 1], mp.variable, mp.tfloat)
        mp.fill(self.zpad, 0.0)
        self.gopad = mp.flux(self.trc, [nb, 1, 1], mp.variable, mp.tfloat)
        mp.fill(self.gopad, 2.0) #start id를 go 토큰으로 사용
        self.endpad = mp.flux(self.trc, [nb, 1, 1], mp.variable, mp.tfloat)
        mp.fill(self.endpad, 0.0) 
        #이하 xsz=16, ubase = 14, latent_sz=32, embede_dim = 64, nb=32, data.py에서 <s> 시작토큰을 없애고 실행한 조건
        

        #xsz=16, latent_sz=128, embede_dim = 128, nb=32, DatasetManager - chatbot, 저장이름 - dual_chatbot
        mp.traceopt(self.trc, 1, 1)
        mp.traceopt(self.trc, 0, 4)
        mp.traceopt(self.trc, 8, 1)
        #mp.traceopt(self.trc, 60, -1) #zstep 망사용 압축 --
        mp.traceopt(self.trc, 69, 777) #rseed
        #mp.traceopt(self.trc, 74, 0) # --
        mp.traceopt(self.trc, 75, 2) #nblock
        #mp.traceopt(self.trc, 76, 1) # --
        mp.traceopt(self.trc, 23, 1e-4)

        #self.in_gate = mp.flux(self.trc, [-1, xsz, 1], mp.variable, mp.tfloat)
        if self.in_plus:
            self.tar_gate = mp.flux(self.trc, [-1, self.seq_len, 1], mp.variable, mp.tfloat)
        else:
            self.tar_gate = mp.flux(self.trc, [-1, self.half_len + 1, 1], mp.variable, mp.tfloat)
        self.by_gate = mp.flux(self.trc, [-1, self.seq_len, 1], mp.variable, mp.tfloat)
        mp.setbygate(self.trc, self.by_gate, self.half_len + 1, self.zpad, self.zpad)
        self.cell = mp.generic(None, self.tar_gate, latent_sz, len(self.dm.source_id2word), 
            len(self.dm.target_id2word), embede_dim)#,  mp.actf_lrelu)
        #self.cell = mp.generic(None, self.tar_gate, latent_sz, 0, 
        #    len(self.dm.target_id2word), embede_dim)#,  mp.actf_lrelu)

        self.accu_input = mp.flux(self.trc, [-1, self.half_len, 1], mp.variable, mp.tfloat)
        self.accu_label = mp.flux(self.trc, [-1, self.half_len, 1], mp.variable, mp.tfloat)
        mp.accuracy(self.cell, self.accu_input, self.accu_label)
        self.open_logf('w')

    def open_logf(self, mode):
        if self.model_name is not None:
            self.logfp = open(self.model_name, mode)

    def logf(self, format, *args):
        data = format % args
        print(data)
        #if self.logfp is not None:
        #    self.logfp.write(data + '\n')
    def logf2(self, format, *args):
        data = format % args
        print(data)
        if self.logfp is not None:
            self.logfp.write(data + '\n')
            self.logfp.flush()
    def close_logf(self):
        if self.logfp is not None: self.logfp.close()

    def predict(self, input_ids):#입쳑 시퀀스로부터 바로 타겟 시퀀스 예측
        mp.copya(self.in_d, input_ids)
        #mp.copya(self.tar_d, sample_y)

        mp.fill(self.by, 0.0)#reset #원래는 전체 리셋하지 않고 밑에서 go token 적재해야함, 테스트로 결과를 확실히 알기위해 b전체 리텟
        mp.howrite(self.by, self.in_d)#input
        mp.howrite(self.by, self.zpad, self.half_len)#padding, 입력과 타겟 사이즈를 동일하게 할경우 필요, 아니면 패딩 필요없음
        mp.howrite(self.by, self.gopad, self.half_len + 1)#go token
        mp.feedf(self.by_gate, self.by)
        #mp.feeda(self.in_gate, input_ids)
        self.y_pred = mp.predict(self.cell)
        return mp.eval(self.y_pred)

    def predicts(self, input_ids):
        y_preds = []
        n = input_ids.shape[0]
        i = 0
        while i + self.batch_size <= n:
            in_batch = input_ids[i:i + self.batch_size]
            y_pred = self.predict(in_batch)
            #print(y_pred)
            y_preds.append(y_pred)
            i += self.batch_size
        return np.array(y_preds, dtype='i').reshape(-1, y_pred.shape[1], y_pred.shape[2]), i
    
    def accuracy(self, input_ids, target_ids):
        # 테스트용데이터로 rmse오차를 구한다
        #print("=======================")
        #print(input_ids)
        predict_res, sz = self.predicts(input_ids)
        predict_res = predict_res[::, self.iaccu_pred:-1].copy()#critical 복사하지 않으면 사이즈 그대로
        #print(predict_res)
        #print("+++++++++++++++++++++++++")
        #print(predict_res)
        target_ids = target_ids[0:sz]
        #print("------------------")
        #print(target_ids)
        mp.copya(self.accu_input, predict_res)
        mp.copya(self.accu_label, target_ids)
        error = mp.measure_accuracy(self.cell)
       
        return mp.eval(error), input_ids, predict_res, target_ids

    def recover_sentence(self, sent_ids, id2word):
        #Convert a list of word ids back to a sentence string.
        #for i in sent_ids:
        #    print(i)
        #    print(id2word[i])
        words = list(map(lambda i: id2word[i] if 0 <= i < len(id2word) else '<unk>', sent_ids))

        # Then remove tailing <pad>
        i = len(words) - 1
        while i >= 0 and words[i] == '<pad>':
            i -= 1
        words = words[:i + 1]
        return ' '.join(words)

    def evaluate(self, msg, input_ids, pred_ids, target_ids):
        """Make a prediction and compute BLEU score.
        """
        refs = []
        hypos = []
        
        pred_ids = np.squeeze(pred_ids)
        target_ids = np.squeeze(target_ids)
        
        self.logf("\n=================== %s ===========================", msg)
        if input_ids is not None:
            input_ids = np.squeeze(input_ids)
            for sor, truth, pred in zip(input_ids, target_ids, pred_ids):
                source_sent = self.recover_sentence(sor, self.dm.source_id2word)
                truth_sent = self.recover_sentence(truth, self.dm.target_id2word)
                pred_sent = self.recover_sentence(pred, self.dm.target_id2word)
                self.logf("[Source] %s", source_sent)
                self.logf("[Truth] %s", truth_sent)
                self.logf("[Translated] %s\n", pred_sent)
        else:
            for truth, pred in zip(target_ids, pred_ids):
                truth_sent = self.recover_sentence(truth, self.dm.target_id2word)
                pred_sent = self.recover_sentence(pred, self.dm.target_id2word)
                self.logf("[Truth] %s", truth_sent)
                self.logf("[Translated] %s\n", pred_sent)
        #smoothie = SmoothingFunction().method4
        #bleu_score = corpus_bleu(refs, hypos, smoothing_function=smoothie)
        #return {'bleu_score': bleu_score * 100.}

    def train(self, n_step=None):
        max_v = 0
        i_step = 0
        i_max = 0
        while n_step is None or i_step < n_step:
            
            try:
                sample_x, sample_y, ep = next(self.train_data_iter)
                #if i_step == 0:
                #    sample_x, sample_y, ep = next(self.train_data_iter)
                mp.copya(self.in_d, sample_x)
                mp.copya(self.tar_d, sample_y)

                mp.howrite(self.by, self.in_d)#input
                mp.howrite(self.by, self.zpad, self.half_len)#padding, 입력과 타겟 사이즈를 동일하게 할경우 필요, 아니면 패딩 필요없음
                mp.howrite(self.by, self.gopad, self.half_len + 1)#go token
                mp.howrite(self.by, self.tar_d, self.half_len + 2)#target
                if self.in_plus:
                    mp.howrite(self.ytar, self.in_d)#input
                    mp.howrite(self.ytar, self.zpad, self.half_len)#padding
                    mp.howrite(self.ytar, self.tar_d, self.half_len + 1)#target
                    mp.howrite(self.ytar, self.endpad, self.half_len + self.half_len + 1)#end token
                else:
                    mp.howrite(self.ytar, self.tar_d)#target
                    mp.howrite(self.ytar, self.endpad, self.half_len)#end token

                mp.feedf(self.by_gate, self.by)
                mp.feedf(self.tar_gate, self.ytar)

                total_loss, _ = mp.train(self.cell)
                #mp.predict(self.cell)
                print('loss: ', mp.eval(total_loss)[0])

            except StopIteration:
                break
            
            if i_step % 400 == 0:
                sample_x, sample_y, _ = next(self.test_batch_iter)
                print(sample_x.shape)
                test_accu, test_input, test_predict, test_right = self.accuracy(sample_x, sample_y)
                self.evaluate('test batch', test_input, test_predict, test_right)

                sample_x, sample_y, _ = next(self.train_batch_iter)
                train_accu, train_input, train_predict, train_right = self.accuracy(sample_x, sample_y)
                self.evaluate('train batch', train_input, train_predict, train_right)
                #self.logf("epoch: %d step: %d, train(A): %f", ep, i_step, train_accu)
                self.logf2("epoch: %d step: %d, train(A): %f, test(B): %f, B-A: %f max: %f imax: %d",
                    ep, i_step, train_accu, test_accu, test_accu-train_accu, max_v, i_max)
                if max_v < train_accu: 
                    max_v = train_accu
                    i_max = i_step
                    mp.save_weight(self.trc)
                if train_accu > 0.98:
                    exit(0)
            if i_step % 1000 == 0: time.sleep(5)
            i_step += 1
        #mp.save_weight(self.trc)

    def notrain(self):
        i_step = 0
        ep = 0
        while ep < 20:
            sample_x, sample_y, _ = next(self.test_batch_iter)
            #print(sample_x.shape)
            test_error, test_input, test_predict, test_right = self.accuracy(sample_x, sample_y)
            self.evaluate('test batch', test_input, test_predict, test_right)

            sample_x, sample_y, _ = next(self.train_batch_iter)
            train_error, train_input, train_predict, train_right = self.accuracy(sample_x, sample_y)
            self.evaluate('train batch', train_input, train_predict, train_right)
            #self.logf("epoch: %d step: %d, train(A): %f", ep, i_step, train_error)
            self.logf("epoch: %d step: %d, train(A): %f, test(B): %f, B-A: %f",
                ep, i_step, train_error, test_error, test_error-train_error)
            print("ep #: ", ep, "step #: ", i_step)
            i_step += 1


kmodel = KModel(name='dual_chatbot') #option
#kmodel.train(50000000)
kmodel.notrain()
#kmodel.evaluate('test', kmodel.test_input, kmodel.test_predict, kmodel.test_right)
#kmodel.evaluate('train', kmodel.train_input, kmodel.train_predict, kmodel.train_right)