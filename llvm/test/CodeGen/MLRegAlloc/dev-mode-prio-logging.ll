; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
;
; Check that we log correctly, both with a learned policy, and the default policy
;
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t1 \
; RUN:   < %S/Inputs/input.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t1 > %t1.readable
; RUN: FileCheck --input-file %t1.readable %s --check-prefixes=CHECK,NOML
; RUN: diff %t1.readable %S/Inputs/reference-prio-log-noml.txt

; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-priority-test-model.py %t_savedmodel
; RUN: %python %S/../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-priority-advisor=development \
; RUN:   -regalloc-priority-training-log=%t2 \
; RUN:   -regalloc-priority-model=%t < %S/Inputs/input.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t2 > %t2.readable
; RUN: FileCheck --input-file %t2.readable %s --check-prefixes=CHECK,ML

; CHECK-NOT: nan
; CHECK-LABEL: priority:
; NOML-SAME: 2684358144.0
; ML-SAME: 3535
; CHECK-LABEL: reward:
