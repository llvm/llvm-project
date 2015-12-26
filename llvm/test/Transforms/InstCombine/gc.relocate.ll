; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; Uses InstCombine with DataLayout to propagate dereferenceable
; attribute via gc.relocate: if the derived ptr is dereferenceable(N),
; then the return attribute of gc.relocate is dereferenceable(N).

declare zeroext i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)

define i32 addrspace(1)* @deref(i32 addrspace(1)* dereferenceable(8) %dparam) gc "statepoint-example" {
; Checks that a dereferenceabler pointer
; CHECK-LABEL: @deref
; CHECK: call dereferenceable(8)
entry:
    %load = load i32, i32 addrspace(1)* %dparam
    %tok = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 7, i32 7)
    ret i32 addrspace(1)* %relocate
}

define i32 @explicit_nonnull(i32 addrspace(1)* nonnull %dparam) gc "statepoint-example" {
; Checks that a nonnull pointer
; CHECK-LABEL: @explicit_nonnull
; CHECK: ret i32 1
entry:
    %load = load i32, i32 addrspace(1)* %dparam
    %tok = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 7, i32 7)
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
    %tok = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32 addrspace(1)* %dparam)
    %relocate = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 7, i32 7)
    %cmp = icmp eq i32 addrspace(1)* %relocate, null
    %ret_val = select i1 %cmp, i32 0, i32 1
    ret i32 %ret_val
no_gc:
    unreachable
}
