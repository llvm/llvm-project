; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s

; Test that basic memory operations assemble as expected with 32-bit addresses.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.wasm.current.memory.i32() nounwind readonly
declare void @llvm.wasm.grow.memory.i32(i32) nounwind

; CHECK-LABEL: current_memory:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: current_memory $push0={{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @current_memory() {
  %a = call i32 @llvm.wasm.current.memory.i32()
  ret i32 %a
}

; CHECK-LABEL: grow_memory:
; CHECK-NEXT: .param i32{{$}}
; CHECK: grow_memory $0{{$}}
; CHECK-NEXT: return{{$}}
define void @grow_memory(i32 %n) {
  call void @llvm.wasm.grow.memory.i32(i32 %n)
  ret void
}
