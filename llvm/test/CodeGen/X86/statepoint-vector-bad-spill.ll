; RUN: llc -verify-machineinstrs -O3 < %s | FileCheck %s

; This is checking for a crash.

target triple = "x86_64-pc-linux-gnu"

define <2 x i8 addrspace(1)*> @test0(i8 addrspace(1)* %el, <2 x i8 addrspace(1)*>* %vec_ptr) gc "statepoint-example" {
; CHECK-LABEL: test0:

entry:
  %tok0 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, i8 addrspace(1)* %el)
  %el.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tok0, i32 7, i32 7)

  %obj.pre = load <2 x i8 addrspace(1)*>, <2 x i8 addrspace(1)*>* %vec_ptr
  %obj = insertelement <2 x i8 addrspace(1)*> %obj.pre, i8 addrspace(1)* %el.relocated, i32 0  ; No real objective here, except to use %el

  %tok1 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, <2 x i8 addrspace(1)*> %obj)
  %obj.relocated = call <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %tok1, i32 7, i32 7)
  ret <2 x i8 addrspace(1)*> %obj.relocated
}

define i8 addrspace(1)* @test1(<2 x i8 addrspace(1)*> %obj) gc "statepoint-example" {
; CHECK-LABEL: test1:

entry:
  %tok1 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, <2 x i8 addrspace(1)*> %obj)
  %obj.relocated = call <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token %tok1, i32 7, i32 7)

  %el = extractelement <2 x i8 addrspace(1)*> %obj.relocated, i32 0
  %tok0 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @do_safepoint, i32 0, i32 0, i32 0, i32 0, i8 addrspace(1)* %el)
  %el.relocated = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tok0, i32 7, i32 7)
  ret i8 addrspace(1)* %el.relocated
}

declare void @do_safepoint()

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
declare <2 x i8 addrspace(1)*> @llvm.experimental.gc.relocate.v2p1i8(token, i32, i32)
