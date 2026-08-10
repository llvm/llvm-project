; RUN: llc -verify-machineinstrs -O3 < %s | FileCheck %s

; This is checking for a crash.

target triple = "x86_64-pc-linux-gnu"

define <2 x ptr addrspace(1)> @test0(ptr addrspace(1) %el, ptr %vec_ptr) gc "statepoint-example" {
; CHECK-LABEL: test0:

entry:
  %tok0 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %el)]
  %el.relocated = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok0, i32 0, i32 0)

  %obj.pre = load <2 x ptr addrspace(1)>, ptr %vec_ptr
  %obj = insertelement <2 x ptr addrspace(1)> %obj.pre, ptr addrspace(1) %el.relocated, i32 0  ; No real objective here, except to use %el

  %tok1 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live" (<2 x ptr addrspace(1)> %obj)]
  %obj.relocated = call <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token %tok1, i32 0, i32 0)
  ret <2 x ptr addrspace(1)> %obj.relocated
}

define ptr addrspace(1) @test1(<2 x ptr addrspace(1)> %obj) gc "statepoint-example" {
; CHECK-LABEL: test1:

entry:
  %tok1 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live" (<2 x ptr addrspace(1)> %obj)]
  %obj.relocated = call <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token %tok1, i32 0, i32 0)

  %el = extractelement <2 x ptr addrspace(1)> %obj.relocated, i32 0
  %tok0 = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @do_safepoint, i32 0, i32 0, i32 0, i32 0) ["gc-live" (ptr addrspace(1) %el)]
  %el.relocated = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok0, i32 0, i32 0)
  ret ptr addrspace(1) %el.relocated
}

declare void @do_safepoint()

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
declare <2 x ptr addrspace(1)> @llvm.experimental.gc.relocate.v2p1(token, i32, i32)
