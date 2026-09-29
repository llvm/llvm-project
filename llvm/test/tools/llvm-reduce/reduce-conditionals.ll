; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=simplify-conditionals-true --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT-TRUE,RESULT,CHECK %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=simplify-conditionals-false --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS,CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT-FALSE,RESULT,CHECK %s < %t

; Make sure there is no unreachable code introduced by the reduction

; CHECK-LABEL: @func_simplifies_true(
; CHECK-INTERESTINGNESS: store i32 1,

; RESULT-TRUE: bb0:
; RESULT-TRUE: store i32 0, ptr null, align 4
; RESULT-TRUE-NEXT: store i32 1, ptr null, align 4
; RESULT-TRUE-NEXT: br label %bb2
; RESULT-TRUE-NOT: bb1

; RESULT-FALSE: bb0:
; RESULT-FALSE-NEXT: store i32 0, ptr null, align 4
; RESULT-FALSE-NEXT: br i1 %cond0, label %bb1, label %bb2

; RESULT-FALSE: bb1:                                              ; preds = %bb0
; RESULT-FALSE-NEXT: store i32 1, ptr null, align 4
; RESULT-FALSE-NEXT: br label %bb3

; RESULT-FALSE: bb2:                                              ; preds = %bb0
; RESULT-FALSE-NEXT: store i32 2, ptr null, align 4
; RESULT-FALSE-NEXT: br label %bb3

; RESULT-FALSE: bb3:                                              ; preds = %bb1, %bb2
; RESULT-FALSE-NEXT: ret void
define void @func_simplifies_true(i1 %cond0, i1 %cond1) {
bb0:
  store i32 0, ptr null
  br i1 %cond0, label %bb1, label %bb2

bb1:
  store i32 1, ptr null
  br i1 %cond1, label %bb2, label %bb3

bb2:
  store i32 2, ptr null
  br label %bb3

bb3:
  ret void
}

; CHECK-LABEL: @func_simplifies_false(
; CHECK-INTERESTINGNESS: store i32 0,

; RESULT-TRUE: bb0:
; RESULT-TRUE: store i32 0, ptr null, align 4
; RESULT-TRUE-NEXT: store i32 1, ptr null, align 4
; RESULT-TRUE-NEXT: br label %bb2
; RESULT-TRUE-NOT: bb1


; RESULT-FALSE: bb0:
; RESULT-FALSE: store i32 0, ptr null, align 4
; RESULT-FALSE-NEXT: br label %bb2

; RESULT-FALSE: bb2: ; preds = %bb0
; RESULT-FALSE-NEXT: store i32 2, ptr null, align 4
; RESULT-FALSE-NEXT: br label %bb3

; RESULT-FALSE: bb3: ; preds = %bb2
; RESULT-FALSE-NEXT: ret void
define void @func_simplifies_false(i1 %cond0, i1 %cond1) {
bb0:
  store i32 0, ptr null
  br i1 %cond0, label %bb1, label %bb2

bb1:
  store i32 1, ptr null
  br i1 %cond1, label %bb2, label %bb3

bb2:
  store i32 2, ptr null
  br label %bb3

bb3:
  ret void
}

; Make sure we don't break the reduction in the other functions by
; having something interesting in unrelated unreachable code.

; CHECK-LABEL: @func_simplifies_true_with_interesting_unreachable_code(
; CHECK-INTERESTINGNESS: store i32 0,
; CHECK-INTERESTINGNESS: store i32 %arg,


; RESULT: bb0:
; RESULT-NEXT: store i32 0
; RESULT-NEXT: br i1 %cond0, label %bb1, label %bb2

; RESULT: bb1:
; RESULT-NEXT: store i32 1
; RESULT-NEXT: br i1 %cond1, label %bb2, label %bb3

; RESULT: bb2:
; RESULT-NEXT: store i32 2
; RESULT-NEXT: br label %bb3

; RESULT: dead_code: ; preds = %dead_code
; RESULT-NEXT: store i32 %arg,
; RESULT-NEXT: br label %dead_code
define void @func_simplifies_true_with_interesting_unreachable_code(i1 %cond0, i1 %cond1, i32 %arg) {
bb0:
  store i32 0, ptr null
  br i1 %cond0, label %bb1, label %bb2

bb1:
  store i32 1, ptr null
  br i1 %cond1, label %bb2, label %bb3

bb2:
  store i32 2, ptr null
  br label %bb3

bb3:
  ret void

dead_code:
  store i32 %arg, ptr null
  br label %dead_code
}

@block_address_user = constant [1 x ptr] [ptr blockaddress(@will_be_unreachable_blockaddress_use, %will_be_unreachable)]

; CHECK-LABEL: @will_be_unreachable_blockaddress_use(
; CHECK-INTERESTINGNESS: inttoptr

; RESULT-FALSE: entry:
; RESULT-FALSE-NEXT: %i2p = inttoptr i64 %int to ptr
; RESULT-FALSE-NEXT: br label %exit

; RESULT-FALSE: exit: ; preds = %entry
; RESULT-FALSE-NEXT: ret i1 false
define i1 @will_be_unreachable_blockaddress_use(i1 %cond, i64 %int) {
entry:
  %i2p = inttoptr i64 %int to ptr
  br i1 %cond, label %will_be_unreachable, label %exit

will_be_unreachable:
  %load = load ptr, ptr %i2p, align 8
  br label %for.body

for.body:
  br label %for.body

exit:
  ret i1 false
}
