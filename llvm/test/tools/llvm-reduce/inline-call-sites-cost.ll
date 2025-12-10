; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=inline-call-sites -reduce-callsite-inline-threshold=3 --test FileCheck --test-arg --check-prefix=CHECK --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=RESULT,CHECK %s < %t

declare void @extern_b()
declare void @extern_a()

; RESULT: @gv_init = global ptr @no_inline_noncall_user
@gv_init = global ptr @no_inline_noncall_user


; CHECK-LABEL: define void @no_inline_noncall_user(
define void @no_inline_noncall_user() {
  call void @extern_a()
  call void @extern_a()
  call void @extern_a()
  call void @extern_a()
  ret void
}

; RESULT-LABEL: define void @noncall_user_call() {
; RESULT-NEXT: call void @no_inline_noncall_user()
; RESULT-NEXT: ret void
define void @noncall_user_call() {
  call void @no_inline_noncall_user()
  ret void
}

; RESULT-LABEL: define void @big_callee_small_caller_callee() {
define void @big_callee_small_caller_callee() {
  call void @extern_a()
  call void @extern_a()
  call void @extern_a()
  call void @extern_a()
  ret void
}

; RESULT-LABEL: define void @big_callee_small_caller_caller() {
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: call void @extern_a()
; RESULT-NEXT: call void @extern_a()
; RESULT-NEXT: call void @extern_a()
; RESULT-NEXT: call void @extern_a()
; RESULT-NEXT: ret void
define void @big_callee_small_caller_caller() {
  call void @extern_b()
  call void @big_callee_small_caller_callee()
  ret void
}

; RESULT-LABEL: define void @small_callee_big_caller_callee() {
; RESULT-NEXT: call void @extern_a()
; RESULT-NEXT: ret void
define void @small_callee_big_caller_callee() {
  call void @extern_a()
  ret void
}

; RESULT-LABEL: define void @small_callee_big_caller_caller() {
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: call void @extern_a()
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: ret void
define void @small_callee_big_caller_caller() {
  call void @extern_b()
  call void @small_callee_big_caller_callee()
  call void @extern_b()
  call void @extern_b()
  ret void
}

; RESULT-LABEL: define void @big_callee_big_caller_callee() {
define void @big_callee_big_caller_callee() {
  call void @extern_a()
  call void @extern_a()
  call void @extern_a()
  call void @extern_a()
  ret void
}

; RESULT-LABEL: define void @big_callee_big_caller_caller() {
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: call void @big_callee_big_caller_callee()
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: call void @extern_b()
; RESULT-NEXT: ret void
define void @big_callee_big_caller_caller() {
  call void @extern_b()
  call void @big_callee_big_caller_callee()
  call void @extern_b()
  call void @extern_b()
  call void @extern_b()
  ret void
}
