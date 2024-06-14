; RUN: llc < %s -mtriple=x86_64-unknown | FileCheck %s

define ptr addrspace(1) @no_extra_const(ptr addrspace(1) %obj) gc "statepoint-example" {
; CHECK-LABEL:   no_extra_const:
; CHECK:	       .cfi_startproc
; CHECK-NEXT:    # %bb.0:                                # %entry
; CHECK-NEXT:    pushq	%rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    movq	%rdi, (%rsp)
; CHECK-NEXT:    nopl	8(%rax)
; CHECK-NEXT:    .Ltmp0:
; CHECK-NEXT:    movq	(%rsp), %rax
; CHECK-NEXT:    popq	%rcx
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
entry:
  %safepoint_token = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 4, ptr elementtype(void ()) null, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %obj)]
  %obj.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %safepoint_token, i32 0, i32 0) ; (%obj, %obj)
  ret ptr addrspace(1) %obj.relocated
}

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
