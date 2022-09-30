; REQUIRES: have_tf_api
; REQUIRES: x86_64-linux
;
; Check that we log correctly, both with a learned policy, and the default policy
;
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t1 -tfutils-text-log < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t1
; RUN: sed -i 's/\\n key:/\n key:/g' %t1
; RUN: sed -i 's/\\n feature/\n feature/g' %t1
; RUN: sed -i 's/\\n/ /g' %t1
; RUN: FileCheck --input-file %t1 %s --check-prefixes=CHECK,NOML
; RUN: diff %t1 %S/Inputs/reference-prio-log-noml.txt

; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-priority-test-model.py %t_savedmodel
; RUN: %python %S/../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t2 -tfutils-text-log -regalloc-priority-model=%t < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t2
; RUN: sed -i 's/\\n key:/\n key:/g' %t2
; RUN: sed -i 's/\\n feature/\n feature/g' %t2
; RUN: sed -i 's/\\n/ /g' %t2
; RUN: FileCheck --input-file %t2 %s --check-prefixes=CHECK,ML

; CHECK-NOT: nan
; CHECK-LABEL: key: \"priority\"
; NOML-NEXT: feature {  float_list {  value: 2.68435814e+09  }  }
; ML-NEXT: feature {  float_list {  value: 3551  }  }
; CHECK-LABEL: key: \"reward\"
