; RUN: opt < %s -instcombine -S | FileCheck %s
; These tests check the optimizations specific to
; pointers being relocated at a statepoint.


declare void @func()

define i1 @test_negative(i32 addrspace(1)* %p) gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %p)]
  %pnew = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %cmp = icmp eq i32 addrspace(1)* %pnew, null
  ret i1 %cmp
; CHECK-LABEL: test_negative
; CHECK: %pnew = call i32 addrspace(1)*
; CHECK: ret i1 %cmp
}

define i1 @test_nonnull(i32 addrspace(1)* nonnull %p) gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %p)]
  %pnew = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %cmp = icmp eq i32 addrspace(1)* %pnew, null
  ret i1 %cmp
; CHECK-LABEL: test_nonnull
; CHECK: ret i1 false
}

define i1 @test_null() gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* null)]
  %pnew = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %cmp = icmp eq i32 addrspace(1)* %pnew, null
  ret i1 %cmp
; CHECK-LABEL: test_null
; CHECK-NOT: %pnew
; CHECK: ret i1 true
}

define i1 @test_undef() gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @func, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* undef)]
  %pnew = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %cmp = icmp eq i32 addrspace(1)* %pnew, null
  ret i1 %cmp
; CHECK-LABEL: test_undef
; CHECK-NOT: %pnew
; CHECK: ret i1 undef
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32) #3
