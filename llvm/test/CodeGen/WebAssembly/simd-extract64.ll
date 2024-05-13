; RUN: llc < %s -mattr=+simd128 -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals

; Regression test for a crash on wasm64 when trying to lower extract_vector_elt
; with a 64 bit constant:
;
; t19: i64 = extract_vector_elt t18, Constant:i64<0>

target triple = "wasm64-unknown-unknown"

define void @foo() {
  store <4 x i32> zeroinitializer, ptr poison, align 16
  %1 = load <4 x i32>, ptr poison, align 16
  %2 = extractelement <4 x i32> %1, i32 0
  %3 = insertelement <2 x i32> undef, i32 %2, i32 0
  %4 = insertelement <2 x i32> %3, i32 poison, i32 1
  store <2 x i32> %4, ptr poison, align 8
  unreachable
}
