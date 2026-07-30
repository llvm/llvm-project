; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s

; Test that basic memory operations assemble as expected with 64-bit addresses.

target triple = "wasm64-unknown-unknown"

declare i64 @llvm.wasm.memory.size.i64(i32) nounwind readonly
declare i64 @llvm.wasm.memory.grow.i64(i32, i64) nounwind

; CHECK-LABEL: memory_size:
; CHECK-NEXT: .functype memory_size () -> (i64){{$}}
; CHECK-NEXT: memory.size $push0=, 0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @memory_size() {
  %a = call i64 @llvm.wasm.memory.size.i64(i32 0)
  ret i64 %a
}

; CHECK-LABEL: memory_grow:
; CHECK-NEXT: .functype memory_grow (i64) -> (i64){{$}}
; CHECK: memory.grow $push0=, 0, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @memory_grow(i64 %n) {
  %a = call i64 @llvm.wasm.memory.grow.i64(i32 0, i64 %n)
  ret i64 %a
}
