; REQUIRES: have_tflite
; REQUIRES: x86_64-linux
;
; Check that we log the currently in development features correctly with both the default
; case and with a learned policy.
;
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t1 \
; RUN:   -regalloc-enable-development-features < %S/Inputs/input.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t1 > %t1.readable
; RUN: FileCheck --input-file %t1.readable %s

; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-eviction-test-model.py %t_savedmodel
; RUN: %python %S/../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t2 -regalloc-model=%t \
; RUN:   -regalloc-enable-development-features < %S/Inputs/input.ll
; RUN: %python %S/../../../lib/Analysis/models/log_reader.py %t2 > %t2.readable
; RUN: FileCheck --input-file %t2.readable %s

; CHECK-NOT: nan
; Check the first five opcodes in the first eviction problem
; Also, the first eviction problem is significantly less than 300 instructions. Check
; that there is a zero value.
; Note: we're regex-ing some of the opcodes to avoid test flakyness.
; CHECK: instructions: 19,{{([0-9]{4})}},13{{([0-9]{2})}},13{{([0-9]{2})}},{{.*}},0,
; Only the candidate virtreg and the 10th LR are included in this problem. Make
; sure the other LRs have values of zero. There are 2700 0s followed by some 1s.
; There's a limit to how many repetitions can be matched.
; CHECK: instructions_mapping: {{(((0,){27}){100})}}
; CHECK-SAME: 1
; Indexing 300 back from where the candidate vr actual resides due to the fact
; that not all the values between the 10th LR and the candidate are zero.
; CHECK-SAME-COUNT-6600: 0,
; CHECK-SAME: 1
; Ensure that we can still go through the mapping matrices for the rest of the
; eviction problems to make sure we haven't hit the end of the matrix above.
; There are a total of 23 eviction problems with this test.
; CHECK-LABEL: observation: 16
; Make sure that we're exporting the mbb_frequencies. Don't actually check
; values due to all values being floating point/liable to change very easily.
; CHECK: mbb_frequencies:
; Make sure that we have the mbb_mapping feature, and that the first couple
; of values are correct.
; CHECK: mbb_mapping: 0,0,0,0,1,1,1
