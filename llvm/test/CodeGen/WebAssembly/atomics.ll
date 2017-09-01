; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt 
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+atomics | FileCheck %s

; Test that atomic loads are assembled properly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: load_i32_atomic:
; CHECK: i32.atomic.load $push[[NUM:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}

define i32 @load_i32_atomic(i32 *%p) {
  %v = load atomic i32, i32* %p seq_cst, align 4
  ret i32 %v
}
