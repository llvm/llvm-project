; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @llvm.wasm.throw(i32, i8*)
declare void @llvm.wasm.rethrow()

; CHECK-LABEL: throw:
; CHECK-NEXT: i32.const $push0=, 0
; CHECK-NEXT: throw 0, $pop0
define void @throw() {
  call void @llvm.wasm.throw(i32 0, i8* null)
  ret void
}

; CHECK-LABEL: rethrow:
; CHECK-NEXT: rethrow 0
define void @rethrow() {
  call void @llvm.wasm.rethrow()
  ret void
}
