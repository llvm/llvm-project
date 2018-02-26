; RUN: llc < %s -mtriple arm64-apple-darwin -aarch64-enable-ldst-opt=false -disable-post-ra -asm-verbose=false | FileCheck %s
; Disable the load/store optimizer to avoid having LDP/STPs and simplify checks.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Check that we don't try to tail-call with an sret-demoted return.

declare i1024 @test_sret() #0

; CHECK-LABEL: _test_call_sret:
; CHECK: mov  x[[CALLERX8NUM:[0-9]+]], x8
; CHECK: mov  x8, sp
; CHECK-NEXT: bl _test_sret
; CHECK: ldr [[CALLERSRET1:q[0-9]+]], [sp]
; CHECK: str [[CALLERSRET1:q[0-9]+]], [x[[CALLERX8NUM]]]
; CHECK: ret
define i1024 @test_call_sret() #0 {
  %a = call i1024 @test_sret()
  ret i1024 %a
}

; CHECK-LABEL: _test_tailcall_sret:
; CHECK: mov  x[[CALLERX8NUM:[0-9]+]], x8
; CHECK: mov  x8, sp
; CHECK-NEXT: bl _test_sret
; CHECK: ldr [[CALLERSRET1:q[0-9]+]], [sp]
; CHECK: str [[CALLERSRET1:q[0-9]+]], [x[[CALLERX8NUM]]]
; CHECK: ret
define i1024 @test_tailcall_sret() #0 {
  %a = tail call i1024 @test_sret()
  ret i1024 %a
}

; CHECK-LABEL: _test_indirect_tailcall_sret:
; CHECK: mov  x[[CALLERX8NUM:[0-9]+]], x8
; CHECK: mov  x8, sp
; CHECK-NEXT: blr x0
; CHECK: ldr [[CALLERSRET1:q[0-9]+]], [sp]
; CHECK: str [[CALLERSRET1:q[0-9]+]], [x[[CALLERX8NUM]]]
; CHECK: ret
define i1024 @test_indirect_tailcall_sret(i1024 ()* %f) #0 {
  %a = tail call i1024 %f()
  ret i1024 %a
}

attributes #0 = { nounwind }
