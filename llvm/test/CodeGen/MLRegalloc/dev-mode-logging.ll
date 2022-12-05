; REQUIRES: have_tf_api
; REQUIRES: x86_64-linux
;
; Check that we log correctly, both with a learned policy, and the default policy
;
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development -regalloc-training-log=%t1 \
; RUN:   -tfutils-text-log < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t1
; RUN: sed -i 's/\\n key:/\n key:/g' %t1
; RUN: sed -i 's/\\n feature/\n feature/g' %t1
; RUN: sed -i 's/\\n/ /g' %t1
; RUN: FileCheck --input-file %t1 %s --check-prefixes=CHECK,NOML
; RUN: diff %t1 %S/Inputs/reference-log-noml.txt

; RUN: rm -rf %t_savedmodel %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-eviction-test-model.py %t_savedmodel
; RUN: %python %S/../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development -regalloc-training-log=%t2 \
; RUN:   -tfutils-text-log -regalloc-model=%t < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t2
; RUN: sed -i 's/\\n key:/\n key:/g' %t2
; RUN: sed -i 's/\\n feature/\n feature/g' %t2
; RUN: sed -i 's/\\n/ /g' %t2
; RUN: FileCheck --input-file %t2 %s --check-prefixes=CHECK,ML

; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development -regalloc-training-log=%t3.log \
; RUN:   -tfutils-use-simplelogger < %S/Inputs/two-large-fcts.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t3.log | FileCheck %s --check-prefixes=CHECK-LOG

; CHECK-NOT: nan
; CHECK-LABEL: key: \"index_to_evict\"
; ML-NEXT:    value: 9
; NOML-NEXT:  value: 12
; CHECK-LABEL: key: \"reward\"
; ML:   value: 37.06
; NOML: value: 36.64
; CHECK-NEXT: feature_list
; CHECK-NEXT: key: \"start_bb_freq_by_max\"

; CHECK-LOG: context: SyFgetsCopy
; CHECK-LOG-NEXT: observation: 0
; CHECK-LOG-NEXT: mask: 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK-LOG: index_to_evict: 12
; CHECK-LOG: observation: 16
; CHECK-LOG: reward: 36.64
; CHECK-LOG: context: SyFgets
; CHECK-LOG-NEXT: observation: 0
; CHECK-LOG-NEXT: mask: 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK-LOG: index_to_evict: 12
; CHECK-LOG: observation: 16
; CHECK-LOG: reward: 36.64
