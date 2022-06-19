; REQUIRES: have_tf_api
; REQUIRES: x86_64-linux
;
; Check that we log correctly, both with a learned policy, and the default policy
;
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t1 -tfutils-text-log < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t1
; RUN: sed -i 's/\\n key:/\n key:/g' %t1
; RUN: sed -i 's/\\n feature/\n feature/g' %t1
; RUN: sed -i 's/\\n/ /g' %t1
; RUN: FileCheck --input-file %t1 %s --check-prefixes=CHECK,NOML
; RUN: diff %t1 %S/Inputs/reference-log-noml.txt

; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-eviction-test-model.py %t
; RUN: llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t2 -tfutils-text-log -regalloc-model=%t < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t2
; RUN: sed -i 's/\\n key:/\n key:/g' %t2
; RUN: sed -i 's/\\n feature/\n feature/g' %t2
; RUN: sed -i 's/\\n/ /g' %t2
; RUN: FileCheck --input-file %t2 %s --check-prefixes=CHECK,ML

; CHECK-NOT: nan
; CHECK-LABEL: key: \"index_to_evict\"
; CHECK-NEXT: value: 9
; ML-NEXT:    value: 9
; NOML-NEXT:  value: 32
; CHECK-LABEL: key: \"reward\"
; ML:   value: 36.90
; NOML: value: 36.64
; CHECK-NEXT: feature_list
; CHECK-NEXT: key: \"start_bb_freq_by_max\"
