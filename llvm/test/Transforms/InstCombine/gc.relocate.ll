; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; Uses InstCombine with DataLayout to propagate dereferenceable
; attribute via gc.relocate: if the derived ptr is dereferenceable(N),
; then the return attribute of gc.relocate is dereferenceable(N).

declare zeroext i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)

define i32 @explicit_nonnull(i32 addrspace(1)* nonnull %dparam) gc "statepoint-example" {
; Checks that a nonnull pointer
; CHECK-LABEL: @explicit_nonnull
; CHECK: ret i32 1
entry:
    %load = load i32, i32 addrspace(1)* %dparam
    %tok = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %dparam)]
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 0, i32 0)
    %cmp = icmp eq i32 addrspace(1)* %relocate, null
    %ret_val = select i1 %cmp, i32 0, i32 1
    ret i32 %ret_val
}

define i32 @implicit_nonnull(i32 addrspace(1)* %dparam) gc "statepoint-example" {
; Checks that a nonnull pointer
; CHECK-LABEL: @implicit_nonnull
; CHECK: ret i32 1
entry:
    %cond = icmp eq i32 addrspace(1)* %dparam, null
    br i1 %cond, label %no_gc, label %gc
gc:
    %load = load i32, i32 addrspace(1)* %dparam
    %tok = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %dparam)]
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 0, i32 0)
    %cmp = icmp eq i32 addrspace(1)* %relocate, null
    %ret_val = select i1 %cmp, i32 0, i32 1
    ret i32 %ret_val
no_gc:
    unreachable
}


; Make sure we don't crash when processing vectors
define <2 x i8 addrspace(1)*> @vector(<2 x i8 addrspace(1)*> %obj) gc "statepoint-example" {
entry:
; CHECK-LABEL: @vector
; CHECK: gc.statepoint
; CHECK: gc.relocate
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live"(<2 x i8 addrspace(1)*> %obj)]
  %obj.relocated = call coldcc <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %safepoint_token, i32 0, i32 0) ; (%obj, %obj)
  ret <2 x i8 addrspace(1)*> %obj.relocated
}

define i32 addrspace(1)* @canonical_base(i32 addrspace(1)* %dparam) gc "statepoint-example" {
; Checks that a nonnull pointer
; CHECK-LABEL: @canonical_base
; CHECK: (token %tok, i32 0, i32 0) ; (%dparam, %dparam)
entry:
  %tok = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %dparam, i32 addrspace(1)* %dparam)]
  %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 0, i32 1)
  ret i32 addrspace(1)* %relocate
}


declare void @do_safepoint()

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
declare <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token, i32, i32)
