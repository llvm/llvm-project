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

define <4 x i8> @trunc_v4i32_to_v4i8(<4 x i32> %x) {
; CHECK-LABEL: function 'trunc_v4i32_to_v4i8'
; CHECK: estimated cost of 3 for instruction:
  %result = trunc <4 x i32> %x to <4 x i8>
  ret <4 x i8> %result
}

define <4 x i8> @trunc_v4i16_to_v4i8(<4 x i16> %x) {
; CHECK-LABEL: function 'trunc_v4i16_to_v4i8'
; CHECK: estimated cost of 2 for instruction:
  %result = trunc <4 x i16> %x to <4 x i8>
  ret <4 x i8> %result
}

define <4 x i16> @trunc_v4i64_to_v4i16(<4 x i64> %x) {
; CHECK-LABEL: function 'trunc_v4i64_to_v4i16'
; CHECK: estimated cost of 4 for instruction:
  %result = trunc <4 x i64> %x to <4 x i16>
  ret <4 x i16> %result
}

define <4 x i8> @trunc_v4i64_to_v4i8(<4 x i64> %x) {
; CHECK-LABEL: function 'trunc_v4i64_to_v4i8'
; CHECK: estimated cost of 5 for instruction:
  %result = trunc <4 x i64> %x to <4 x i8>
  ret <4 x i8> %result
}

define <8 x i8> @trunc_v8i32_to_v8i8(<8 x i32> %x) {
; CHECK-LABEL: function 'trunc_v8i32_to_v8i8'
; CHECK: estimated cost of 4 for instruction:
  %result = trunc <8 x i32> %x to <8 x i8>
  ret <8 x i8> %result
}

define <8 x i16> @trunc_v8i32_to_v8i16(<8 x i32> %x) {
; CHECK-LABEL: function 'trunc_v8i32_to_v8i16'
; CHECK: estimated cost of 3 for instruction:
  %result = trunc <8 x i32> %x to <8 x i16>
  ret <8 x i16> %result
}

define <8 x i8> @trunc_v8i64_to_v8i8(<8 x i64> %x) {
; CHECK-LABEL: function 'trunc_v8i64_to_v8i8'
; CHECK: estimated cost of 8 for instruction:
  %result = trunc <8 x i64> %x to <8 x i8>
  ret <8 x i8> %result
}

define <16 x i16> @trunc_v16i32_to_v16i16(<16 x i32> %x) {
; CHECK-LABEL: function 'trunc_v16i32_to_v16i16'
; CHECK: estimated cost of 6 for instruction:
  %result = trunc <16 x i32> %x to <16 x i16>
  ret <16 x i16> %result
}

define <16 x i16> @trunc_v16i64_to_v16i16(<16 x i64> %x) {
; CHECK-LABEL: function 'trunc_v16i64_to_v16i16'
; CHECK: estimated cost of 14 for instruction:
  %result = trunc <16 x i64> %x to <16 x i16>
  ret <16 x i16> %result
}

define <4 x i8> @fptosi_v4f32_to_v4i8(<4 x float> %x) {
; CHECK-LABEL: function 'fptosi_v4f32_to_v4i8'
; CHECK: estimated cost of 5 for instruction:
  %result = fptosi <4 x float> %x to <4 x i8>
  ret <4 x i8> %result
}

define <2 x i8> @fptosi_v2f32_to_v2i8(<2 x float> %x) {
; CHECK-LABEL: function 'fptosi_v2f32_to_v2i8'
; CHECK: estimated cost of 5 for instruction:
  %result = fptosi <2 x float> %x to <2 x i8>
  ret <2 x i8> %result
}

define <2 x i8> @fptoui_v2f32_to_v2i8(<2 x float> %x) {
; CHECK-LABEL: function 'fptoui_v2f32_to_v2i8'
; CHECK: estimated cost of 5 for instruction:
  %result = fptoui <2 x float> %x to <2 x i8>
  ret <2 x i8> %result
}

define <4 x i8> @fptoui_v4f32_to_v4i8(<4 x float> %x) {
; CHECK-LABEL: function 'fptoui_v4f32_to_v4i8'
; CHECK: estimated cost of 5 for instruction:
  %result = fptoui <4 x float> %x to <4 x i8>
  ret <4 x i8> %result
}

define <8 x i8> @fptosi_v8f32_to_v8i8(<8 x float> %x) {
; CHECK-LABEL: function 'fptosi_v8f32_to_v8i8'
; CHECK: estimated cost of 7 for instruction:
  %result = fptosi <8 x float> %x to <8 x i8>
  ret <8 x i8> %result
}

define <8 x i8> @fptoui_v8f32_to_v8i8(<8 x float> %x) {
; CHECK-LABEL: function 'fptoui_v8f32_to_v8i8'
; CHECK: estimated cost of 7 for instruction:
  %result = fptoui <8 x float> %x to <8 x i8>
  ret <8 x i8> %result
}
