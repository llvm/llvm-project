; REQUIRES: have_tf_api
; REQUIRES: x86_64-linux
;
; Check that we log the currently in development features correctly with both the default
; case and with a learned policy.
;
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t1 -tfutils-text-log \
; RUN:   -regalloc-enable-development-features < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t1
; RUN: sed -i 's/\\n key:/\n key:/g' %t1
; RUN: sed -i 's/\\n feature/\n feature/g' %t1
; RUN: sed -i 's/\\n/ /g' %t1
; RUN: FileCheck --input-file %t1 %s

; RUN: rm -rf %t && mkdir %t
; RUN: %python %S/../../../lib/Analysis/models/gen-regalloc-eviction-test-model.py %t_savedmodel
; RUN: %python %S/../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
; RUN: llc -o /dev/null -mtriple=x86_64-linux-unknown -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=development \
; RUN:   -regalloc-training-log=%t2 -tfutils-text-log -regalloc-model=%t \
; RUN:   -regalloc-enable-development-features < %S/Inputs/input.ll
; RUN: sed -i 's/ \+/ /g' %t2
; RUN: sed -i 's/\\n key:/\n key:/g' %t2
; RUN: sed -i 's/\\n feature/\n feature/g' %t2
; RUN: sed -i 's/\\n/ /g' %t2
; RUN: FileCheck --input-file %t2 %s

; CHECK-NOT: nan
; CHECK-LABEL: key: \"instructions\"
; Check the first five opcodes in the first eviction problem
; CHECK-NEXT: value: 19
; CHECK-SAME: value: {{([0-9]{4})}}
; CHECK-SAME: value: 12{{([0-9]{2})}}
; CHECK-SAME: value: 12{{([0-9]{2})}}
; The first eviction problem is significantly less than 300 instructions. Check
; that there is a zero value
; CHECK-SAME: value: 0
; Only the candidate virtreg and the 10th LR are included in this problem. Make
; sure the other LRs have values of zero.
; CHECK-LABEL: key: \"instructions_mapping\"
; CHECK-COUNT-2700: value: 0
; CHECK-SAME: value: 1
; Indexing 300 back from where the candidate vr actual resides due to the fact
; that not all the values between the 10th LR and the candidate are zero.
; CHECK-COUNT-6600: value: 0
; CHECK-SAME: value: 1
; Ensure that we can still go through the mapping matrices for the rest of the
; eviction problems to make sure we haven't hit the end of the matrix above.
; There are a total of 23 eviction problems with this test.
; CHECK-COUNT-15: int64_list
; CHECK: key: \"is_free\"
; Make sure that we're exporting the mbb_frequencies. Don't actually check
; values due to all values being floating point/liable to change very easily.
; CHECK: key: \"mbb_frequencies\"
; Make sure that we have the mbb_mapping feature, and that the first couple
; of values are correct.
; CHECK: key: \"mbb_mapping\"
; CHECK-NEXT: 0
; CHECK-SAME: 0
; CHECK-SAME: 0
; CHECK-SAME: 0
; CHECK-SAME: 0
; CHECK-SAME: 1
; CHECK-SAME: 1
