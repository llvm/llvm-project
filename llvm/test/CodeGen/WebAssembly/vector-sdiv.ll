; RUN: llc < %s -asm-verbose=false -fast-isel=false -disable-wasm-fallthrough-return-opt | FileCheck %s

target triple = "wasm32-unknown-unknown"

; This should be treated as a non-splat vector of pow2 divisor, so sdivs will be
; transformed to shrs in DAGCombiner. There will be 4 stores and 3 shrs (For '1'
; entry we don't need a shr).

; CHECK-LABEL: vector_sdiv:
; CHECK-DAG:  i32.store
; CHECK-DAG:  i32.shr_u
; CHECK-DAG:  i32.store
; CHECK-DAG:  i32.shr_u
; CHECK-DAG:  i32.store
; CHECK-DAG:  i32.shr_u
; CHECK-DAG:  i32.store
define void @vector_sdiv(<4 x i32>* %x, <4 x i32>* readonly %y) {
entry:
  %0 = load <4 x i32>, <4 x i32>* %y, align 16
  %div = sdiv <4 x i32> %0, <i32 1, i32 4, i32 2, i32 8>
  store <4 x i32> %div, <4 x i32>* %x, align 16
  ret void
}
