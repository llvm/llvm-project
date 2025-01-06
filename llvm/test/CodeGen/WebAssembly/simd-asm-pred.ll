; RUN: llc < %s -verify-machineinstrs -mattr=+relaxed-simd | FileCheck %s

; Test that setting "relaxed-simd" target feature set also implies 'simd128' in
; AssemblerPredicate, which is used to verify instructions in AsmPrinter.

target triple = "wasm32-unknown-unknown"

declare <2 x i64> @llvm.wasm.relaxed.laneselect.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)

; The compiled result of this function uses LOCAL_GET_V128, which is predicated
; on the 'simd128' feature. We should be able to compile this when only
; 'relaxed-simd' is set, which implies 'simd128'.
define <2 x i64> @test(<2 x i64>, <2 x i64>, <2 x i64>) #0 {
; CHECK-LABEL: test:
; CHECK:         .functype  test (v128, v128, v128) -> (v128)
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    local.get  0
; CHECK-NEXT:    local.get  1
; CHECK-NEXT:    local.get  2
; CHECK-NEXT:    i64x2.relaxed_laneselect
start:
  %_4 = tail call <2 x i64> @llvm.wasm.relaxed.laneselect.v2i64(<2 x i64> %0, <2 x i64> %1, <2 x i64> %2) #3
  ret <2 x i64> %_4
}
