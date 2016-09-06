; RUN: llc < %s | FileCheck %s

; Checks for a crash we had when two gc.relocate calls would
; relocating identical values

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @f()
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32) #3
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32) #3

define void @test_gcrelocate_uniqueing(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test_gcrelocate_uniqueing
  %tok = tail call token (i64, i32, void ()*, i32, i32, ...)
      @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 0, i32 0, i32 0, i32 2, i32 addrspace(1)* %ptr, i32 undef, i32 addrspace(1)* %ptr, i32 addrspace(1)* %ptr)
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 9, i32 9)
  %b = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 10, i32 10)
  ret void
}

define void @test_gcptr_uniqueing(i32 addrspace(1)* %ptr) gc "statepoint-example" {
; CHECK-LABEL: test_gcptr_uniqueing
  %ptr2 = bitcast i32 addrspace(1)* %ptr to i8 addrspace(1)*
  %tok = tail call token (i64, i32, void ()*, i32, i32, ...)
      @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @f, i32 0, i32 0, i32 0, i32 2, i32 addrspace(1)* %ptr, i32 undef, i32 addrspace(1)* %ptr, i8 addrspace(1)* %ptr2)
  %a = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok, i32 9, i32 9)
  %b = call i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %tok, i32 10, i32 10)
  ret void
}
