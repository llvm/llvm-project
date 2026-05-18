; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=simplify-conditionals-false --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT-FALSE,CHECK %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=simplify-conditionals-true --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT-TRUE,CHECK %s < %t

; Check that simplify-conditionals-true/false do not attempt block
; simplification in cases that happened to already use a constant
; true/false branch. We should not get the side effect of running
; simplifycfg on blocks where we did not change the terminator value,
; and not introduce unreachable code.


; CHECK-LABEL: @br_false(
; RESULT-FALSE: br i1 false, label %will_be_unreachable, label %exit
; RESULT-TRUE: br label %will_be_unreachable
define i1 @br_false(i64 %int) {
entry:
  %i2p = inttoptr i64 %int to ptr
  br i1 false, label %will_be_unreachable, label %exit

will_be_unreachable:
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}

; CHECK-LABEL: @br_false_keep_in_unreachable(
; CHECK: entry
; INTERESTING: [[I2P:%.+]] = inttoptr i64 %int to ptr
; INTERESTING: load ptr, ptr [[I2P]]

; RESULT-FALSE: br i1 false, label %will_be_unreachable, label %exit
; RESULT-TRUE: br label %will_be_unreachable
define i1 @br_false_keep_in_unreachable(i64 %int) {
entry:
  br i1 false, label %will_be_unreachable, label %exit

will_be_unreachable:
  %i2p = inttoptr i64 %int to ptr
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}

; CHECK-LABEL: @br_true(

; RESULT-FALSE: br label %will_be_unreachable
; RESULT-TRUE: br i1 true, label %exit, label %will_be_unreachable
define i1 @br_true(i64 %int) {
entry:
  %i2p = inttoptr i64 %int to ptr
  br i1 true, label %exit, label %will_be_unreachable

will_be_unreachable:
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}

; CHECK-LABEL: @br_true_keep_in_unreachable(
; CHECK: entry:
; INTERESTING: [[I2P:%.+]] = inttoptr i64 %int to ptr
; INTERESTING: load ptr, ptr [[I2P]]

; RESULT-FALSE: br label %will_be_unreachable
; RESULT-TRUE: br i1 true, label %exit, label %will_be_unreachable
define i1 @br_true_keep_in_unreachable(i64 %int) {
entry:
  %i2p = inttoptr i64 %int to ptr
  br i1 true, label %exit, label %will_be_unreachable

will_be_unreachable:
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}

; CHECK-LABEL: @br_poison(
; RESULT-FALSE: br label %will_be_unreachable
; RESULT-TRUE: br label %exit
define i1 @br_poison(i64 %int) {
entry:
  %i2p = inttoptr i64 %int to ptr
  br i1 poison, label %exit, label %will_be_unreachable

will_be_unreachable:
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}

; CHECK-LABEL: @br_poison_keep_in_unreachable(
; CHECK: entry:
; INTERESTING: [[I2P:%.+]] = inttoptr i64 %int to ptr
; INTERESTING: load ptr, ptr [[I2P]]

; RESULT-FALSE: br label %will_be_unreachable
; RESULT-TRUE: br label %exit
define i1 @br_poison_keep_in_unreachable(i64 %int) {
entry:
  %i2p = inttoptr i64 %int to ptr
  br i1 poison, label %exit, label %will_be_unreachable

will_be_unreachable:
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}
