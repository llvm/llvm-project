; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals | FileCheck %s

; Test that basic memory operations assemble as expected with 32-bit addresses.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.wasm.mem.size.i32(i32) nounwind readonly
declare i32 @llvm.wasm.mem.grow.i32(i32, i32) nounwind
declare i32 @llvm.wasm.current.memory.i32() nounwind readonly
declare i32 @llvm.wasm.grow.memory.i32(i32) nounwind

; CHECK-LABEL: mem_size:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: mem.size $push0=, 0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @mem_size() {
  %a = call i32 @llvm.wasm.mem.size.i32(i32 0)
  ret i32 %a
}

; CHECK-LABEL: mem_grow:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK: mem.grow $push0=, 0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @mem_grow(i32 %n) {
  %a = call i32 @llvm.wasm.mem.grow.i32(i32 0, i32 %n)
  ret i32 %a
}

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
; CHECK-NEXT: .result i32{{$}}
; CHECK: grow_memory $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @grow_memory(i32 %n) {
  %a = call i32 @llvm.wasm.grow.memory.i32(i32 %n)
  ret i32 %a
}
