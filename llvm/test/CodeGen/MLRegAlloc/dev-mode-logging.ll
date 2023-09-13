; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
;
; Check that we log correctly, both with a learned policy, and the default policy
;
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development -regalloc-training-log=%t1 < %S/Inputs/input.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t1 > %t1.readable
; RUN: FileCheck --input-file %t1.readable %s --check-prefixes=CHECK,NOML
; RUN: diff %t1.readable %S/Inputs/reference-log-noml.txt

; RUN: rm -rf %t_savedmodel %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-eviction-test-model.py %t_savedmodel
; RUN: %python %S/../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development -regalloc-training-log=%t2 \
; RUN:   -regalloc-model=%t < %S/Inputs/input.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t2 > %t2.readable
; RUN: FileCheck --input-file %t2.readable %s --check-prefixes=CHECK,ML

; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development -regalloc-training-log=%t3.log < %S/Inputs/two-large-fcts.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t3.log | FileCheck %s --check-prefixes=CHECK-TWO-FCTS

; CHECK-NOT: nan
; CHECK-LABEL: context: SyFgets
; CHECK-NEXT: observation: 0
; ML: index_to_evict: 9
; NOML: index_to_evict: 11
; CHECK-NEXT: reward: 0
; CHECK-NEXT: observation: 1
; CHECK-NEXT: mask:
; NOML:      observation: 17
; ML:      observation: 83
; ML: reward: 38.56
; NOML: reward: 37.32


; CHECK-TWO-FCTS: context: SyFgets
; CHECK-TWO-FCTS-NEXT: observation: 0
; CHECK-TWO-FCTS-NEXT: mask: 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK-TWO-FCTS: index_to_evict: 11
; CHECK-TWO-FCTS: observation: 17
; CHECK-TWO-FCTS: reward: 37.32
; CHECK-TWO-FCTS: context: SyFgetsCopy
; CHECK-TWO-FCTS-NEXT: observation: 0
; CHECK-TWO-FCTS-NEXT: mask: 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
; CHECK-TWO-FCTS: index_to_evict: 11
; CHECK-TWO-FCTS: observation: 17
; CHECK-TWO-FCTS: reward: 37.32
