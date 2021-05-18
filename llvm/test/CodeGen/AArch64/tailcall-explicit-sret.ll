; RUN: llc < %s -mtriple arm64-apple-darwin -aarch64-enable-ldst-opt=false -asm-verbose=false -disable-post-ra | FileCheck %s
; Disable the load/store optimizer to avoid having LDP/STPs and simplify checks.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Check that we don't try to tail-call with a non-forwarded sret parameter.
declare void @test_explicit_sret(i1024* sret(i1024)) #0

; This is the only OK case, where we forward the explicit sret pointer.

; CHECK-LABEL: _test_tailcall_explicit_sret:
; CHECK-NEXT: b _test_explicit_sret
define void @test_tailcall_explicit_sret(i1024* sret(i1024) %arg) #0 {
  tail call void @test_explicit_sret(i1024* %arg)
  ret void
}

; CHECK-LABEL: _test_call_explicit_sret:
; CHECK-NOT: mov  x8
; CHECK: bl _test_explicit_sret
; CHECK: ret
define void @test_call_explicit_sret(i1024* sret(i1024) %arg) #0 {
  call void @test_explicit_sret(i1024* %arg)
  ret void
}

; CHECK-LABEL: _test_tailcall_explicit_sret_alloca_unused:
; CHECK: mov  x8, sp
; CHECK-NEXT: bl _test_explicit_sret
; CHECK: ret
define void @test_tailcall_explicit_sret_alloca_unused() #0 {
  %l = alloca i1024, align 8
  tail call void @test_explicit_sret(i1024* %l)
  ret void
}

; CHECK-LABEL: _test_tailcall_explicit_sret_alloca_dummyusers:
; CHECK: ldr [[PTRLOAD1:q[0-9]+]], [x0]
; CHECK: str [[PTRLOAD1]], [sp]
; CHECK: mov  x8, sp
; CHECK-NEXT: bl _test_explicit_sret
; CHECK: ret
define void @test_tailcall_explicit_sret_alloca_dummyusers(i1024* %ptr) #0 {
  %l = alloca i1024, align 8
  %r = load i1024, i1024* %ptr, align 8
  store i1024 %r, i1024* %l, align 8
  tail call void @test_explicit_sret(i1024* %l)
  ret void
}

; This is too conservative, but doesn't really happen in practice.

; CHECK-LABEL: _test_tailcall_explicit_sret_gep:
; CHECK: add  x8, x0, #128
; CHECK-NEXT: bl _test_explicit_sret
; CHECK: ret
define void @test_tailcall_explicit_sret_gep(i1024* %ptr) #0 {
  %ptr2 = getelementptr i1024, i1024* %ptr, i32 1
  tail call void @test_explicit_sret(i1024* %ptr2)
  ret void
}

; CHECK-LABEL: _test_tailcall_explicit_sret_alloca_returned:
; CHECK: mov  x[[CALLERX8NUM:[0-9]+]], x8
; CHECK: mov  x8, sp
; CHECK-NEXT: bl _test_explicit_sret
; CHECK-NEXT: ldr [[CALLERSRET1:q[0-9]+]], [sp]
; CHECK: str [[CALLERSRET1:q[0-9]+]], [x[[CALLERX8NUM]]]
; CHECK: ret
define i1024 @test_tailcall_explicit_sret_alloca_returned() #0 {
  %l = alloca i1024, align 8
  tail call void @test_explicit_sret(i1024* %l)
  %r = load i1024, i1024* %l, align 8
  ret i1024 %r
}

; CHECK-LABEL: _test_indirect_tailcall_explicit_sret_nosret_arg:
; CHECK-DAG: mov  x[[CALLERX8NUM:[0-9]+]], x8
; CHECK-DAG: mov  [[FPTR:x[0-9]+]], x0
; CHECK: mov  x0, sp
; CHECK-NEXT: blr [[FPTR]]
; CHECK: ldr [[CALLERSRET1:q[0-9]+]], [sp]
; CHECK: str [[CALLERSRET1:q[0-9]+]], [x[[CALLERX8NUM]]]
; CHECK: ret
define void @test_indirect_tailcall_explicit_sret_nosret_arg(i1024* sret(i1024) %arg, void (i1024*)* %f) #0 {
  %l = alloca i1024, align 8
  tail call void %f(i1024* %l)
  %r = load i1024, i1024* %l, align 8
  store i1024 %r, i1024* %arg, align 8
  ret void
}

; CHECK-LABEL: _test_indirect_tailcall_explicit_sret_:
; CHECK: mov  x[[CALLERX8NUM:[0-9]+]], x8
; CHECK: mov  x8, sp
; CHECK-NEXT: blr x0
; CHECK: ldr [[CALLERSRET1:q[0-9]+]], [sp]
; CHECK: str [[CALLERSRET1:q[0-9]+]], [x[[CALLERX8NUM]]]
; CHECK: ret
define void @test_indirect_tailcall_explicit_sret_(i1024* sret(i1024) %arg, i1024 ()* %f) #0 {
  %ret = tail call i1024 %f()
  store i1024 %ret, i1024* %arg, align 8
  ret void
}

attributes #0 = { nounwind }
