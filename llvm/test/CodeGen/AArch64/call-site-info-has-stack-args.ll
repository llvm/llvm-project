; RUN: llc -mtriple=aarch64-apple-darwin -emit-codegen-call-site-info < %s -stop-before=finalize-isel -o - | FileCheck %s

; Verify that HasStackArguments is correctly set in CallSiteInfo.

declare void @no_stack_args(i32, i32)
declare void @has_stack_args(i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare void @vararg(i32, ...)

; CHECK-LABEL: name: test_no_stack_args
; CHECK: callSites:
; CHECK:   hasStackArguments:
; CHECK-NEXT: no
define void @test_no_stack_args() {
  call void @no_stack_args(i32 1, i32 2)
  ret void
}

; CHECK-LABEL: name: test_has_stack_args
; CHECK: callSites:
; CHECK:   hasStackArguments:
; CHECK-NEXT: yes
define void @test_has_stack_args() {
  call void @has_stack_args(i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8)
  ret void
}

; CHECK-LABEL: name: test_vararg
; CHECK: callSites:
; CHECK:   hasStackArguments:
; CHECK-NEXT: yes
define void @test_vararg() {
  call void (i32, ...) @vararg(i32 1, i32 2, i32 3)
  ret void
}

; CHECK-LABEL: name: test_vararg_no_stack_args
; CHECK: callSites:
; CHECK:   hasStackArguments:
; CHECK-NEXT: no
define void @test_vararg_no_stack_args() {
  call void (i32, ...) @vararg(i32 1)
  ret void
}

; A sibling tail call leaves its stack arguments in the caller's incoming
; argument area and allocates no outgoing stack itself, but it still passes
; arguments on the stack.

; CHECK-LABEL: name: test_tail_call_stack_args
; CHECK: callSites:
; CHECK:   hasStackArguments:
; CHECK-NEXT: yes
; CHECK: TCRETURNdi @has_stack_args
define void @test_tail_call_stack_args(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i) {
  tail call void @has_stack_args(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i)
  ret void
}

; CHECK-LABEL: name: test_tail_call_no_stack_args
; CHECK: callSites:
; CHECK:   hasStackArguments:
; CHECK-NEXT: no
; CHECK: TCRETURNdi @no_stack_args
define void @test_tail_call_no_stack_args(i32 %a, i32 %b) {
  tail call void @no_stack_args(i32 %a, i32 %b)
  ret void
}
