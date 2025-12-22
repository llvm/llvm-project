; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments-to-return --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

@gv = global i32 0

; INTERESTING-LABEL: @move_entry_block_use_argument_to_return(i32 %arg) {
; INTERESTING: i32 %arg

; RESULT-LABEL: define i32 @move_entry_block_use_argument_to_return(
; RESULT-NEXT: ret i32 %arg
; RESULT-NEXT: }
define void @move_entry_block_use_argument_to_return(i32 %arg) {
  store i32 %arg, ptr @gv
  ret void
}

; INTERESTING-LABEL: @move_entry_block_use_argument_to_return_existing_ret(i32 %arg) {
; INTERESTING: %arg

; RESULT-LABEL: define i32 @move_entry_block_use_argument_to_return_existing_ret(
; RESULT-NEXT: ret i32 %arg
; RESULT-NEXT: }
define i32 @move_entry_block_use_argument_to_return_existing_ret(i32 %arg) {
  store i32 %arg, ptr @gv
  ret i32 0
}

; INTERESTING-LABEL: @move_phi_block_use_argument_to_return(i32 %arg, ptr %ptr0, ptr %ptr1, i1 %cond0, i1 %cond1) {
; INTERESTING: %arg

; RESULT-LABEL: define i32 @move_phi_block_use_argument_to_return(
; RESULT-NEXT: entry:
; RESULT-NEXT: ret i32 %arg
define void @move_phi_block_use_argument_to_return(i32 %arg, ptr %ptr0, ptr %ptr1, i1 %cond0, i1 %cond1) {
entry:
  br i1 %cond0, label %bb0, label %bb1

bb0:
  %phi = phi i32 [ %arg, %entry ], [ 123, %bb1 ]
  store i32 %arg, ptr %ptr0
  store i32 %phi, ptr %ptr1
  br label %bb1

bb1:
  br i1 %cond1, label %bb0, label %bb2

bb2:
  ret void
}

; INTERESTING-LABEL: define {{.*}} @keep_first_arg(i32 %arg0, ptr %arg1) {
; INTERESTING: %arg0

; RESULT-LABEL: define i32 @keep_first_arg(
; RESULT-NEXT: ret i32 %arg0
; RESULT-NEXT: }
define void @keep_first_arg(i32 %arg0, ptr %arg1) {
  store i32 %arg0, ptr %arg1
  ret void
}

; INTERESTING-LABEL: define {{.*}} @keep_second_arg(i32 %arg0, ptr %arg1) {
; INTERESTING: %arg1

; RESULT-LABEL: define ptr @keep_second_arg(
; RESULT-NEXT: ret ptr %arg1
; RESULT-NEXT: }
define void @keep_second_arg(i32 %arg0, ptr %arg1) {
  store i32 %arg0, ptr %arg1
  ret void
}

; INTERESTING-LABEL: @multi_void_return_arg(i1 %arg0, ptr %arg1, i32 %arg2) {
; INTERESTING: i32 %arg2

; RESULT-LABEL: define i32 @multi_void_return_arg(i1 %arg0, ptr %arg1, i32 %arg2) {
; RESULT-NEXT: entry:
; RESULT-NEXT: ret i32 %arg2
define void @multi_void_return_arg(i1 %arg0, ptr %arg1, i32 %arg2) {
entry:
  br i1 %arg0, label %bb0, label %bb1

bb0:
  store i32 %arg2, ptr %arg1
  ret void

bb1:
  ret void
}
