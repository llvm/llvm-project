; RUN: not opt -S < %s -verify 2>&1 | FileCheck %s

; CHECK: gc.statepoint: number of deoptimization arguments must be a constant integer

declare void @use(...)
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32)
declare i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token, i32, i32)
declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
declare i32 @"personality_function"()

;; Basic usage
define i64 addrspace(1)* @test1(i8 addrspace(1)* %arg, i32 %val) gc "statepoint-example" {
entry:
  %cast = bitcast i8 addrspace(1)* %arg to i64 addrspace(1)*
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 %val, i32 0, i32 0, i32 0, i32 10, i32 0, i8 addrspace(1)* %arg, i64 addrspace(1)* %cast, i8 addrspace(1)* %arg, i8 addrspace(1)* %arg)
  %reloc = call i64 addrspace(1)* @llvm.experimental.gc.relocate.p1i64(token %safepoint_token, i32 12, i32 13)
  ret i64 addrspace(1)* %reloc
}
