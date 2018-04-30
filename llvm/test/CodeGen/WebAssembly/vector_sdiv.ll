; RUN: llc < %s -asm-verbose=false -fast-isel=false -disable-wasm-fallthrough-return-opt | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-elf"

; Because there is a 1 in the vector, sdiv should not be reduced to shifts.

; CHECK-LABEL: vector_sdiv:
; CHECK-DAG:  i32.store
; CHECK-DAG:  i32.div_s
; CHECK-DAG:  i32.store
; CHECK-DAG:  i32.div_s
; CHECK-DAG:  i32.store
; CHECK-DAG:  i32.div_s
; CHECK-DAG:  i32.store
define void @vector_sdiv(<4 x i32>* %x, <4 x i32>* readonly %y) {
entry:
  %0 = load <4 x i32>, <4 x i32>* %y, align 16
  %div = sdiv <4 x i32> %0, <i32 1, i32 4, i32 2, i32 8>
  store <4 x i32> %div, <4 x i32>* %x, align 16
  ret void
}
