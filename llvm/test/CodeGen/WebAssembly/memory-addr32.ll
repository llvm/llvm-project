; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that basic memory operations assemble as expected with 32-bit addresses.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.wasm.memory.size.i32(i32) nounwind readonly
declare i32 @llvm.wasm.memory.grow.i32(i32, i32) nounwind

; CHECK-LABEL: memory_size:
; CHECK-NEXT: .functype memory_size () -> (i32){{$}}
; CHECK-NEXT: memory.size $push0=, 0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @memory_size() {
  %a = call i32 @llvm.wasm.memory.size.i32(i32 0)
  ret i32 %a
}

; CHECK-LABEL: memory_grow:
; CHECK-NEXT: .functype memory_grow (i32) -> (i32){{$}}
; CHECK: memory.grow $push0=, 0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @memory_grow(i32 %n) {
  %a = call i32 @llvm.wasm.memory.grow.i32(i32 0, i32 %n)
  ret i32 %a
}
