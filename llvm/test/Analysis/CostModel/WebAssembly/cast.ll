; RUN: opt -mtriple=wasm32-unknown-unknown -mattr=+simd128 \
; RUN:   -passes='print<cost-model>' -disable-output < %s 2>&1 | FileCheck %s

define <8 x float> @sitofp_v8i16(<8 x i16> %x) {
; CHECK-LABEL: function 'sitofp_v8i16'
; CHECK: estimated cost of 10 for instruction:
  %result = sitofp <8 x i16> %x to <8 x float>
  ret <8 x float> %result
}

define <8 x float> @uitofp_v8i16(<8 x i16> %x) {
; CHECK-LABEL: function 'uitofp_v8i16'
; CHECK: estimated cost of 10 for instruction:
  %result = uitofp <8 x i16> %x to <8 x float>
  ret <8 x float> %result
}
