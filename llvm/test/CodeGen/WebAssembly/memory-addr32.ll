; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic memory operations assemble as expected with 32-bit addresses.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @llvm.wasm.memory.size.i32() nounwind readnone
declare void @llvm.wasm.grow.memory.i32(i32) nounwind

; CHECK-LABEL: memory_size:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: memory_size
; CHECK-NEXT: set_local 0, $pop{{$}}
; CHECK-NEXT: return (get_local 0){{$}}
define i32 @memory_size() {
  %a = call i32 @llvm.wasm.memory.size.i32()
  ret i32 %a
}

; CHECK-LABEL: grow_memory:
; CHECK-NEXT: .param i32
; CHECK-NEXT: .local i32{{$}}
; CHECK: grow_memory (get_local 0)
; CHECK-NEXT: return
define void @grow_memory(i32 %n) {
  call void @llvm.wasm.grow.memory.i32(i32 %n)
  ret void
}
