; RUN: opt -S -passes=vector-combine -data-layout=E -mtriple=spirv-unknown-vulkan1.3-library %s | FileCheck %s

; Ensure the loading of a splatted single scalar doesn't get widened.

define <4 x float> @load_sf32_v4f32(ptr dereferenceable(16) %p) {
; CHECK-LABEL: @load_sf32_v4f32(
; CHECK-NEXT:  [[L:%.*]] = load float, ptr [[P:%.*]], align 4
; CHECK-NEXT:  [[SP:%.*]] = insertelement <4 x float> poison, float [[L]], i64 0
; CHECK-NEXT:  [[SH:%.*]] = shufflevector <4 x float> [[SP]], <4 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:  ret <4 x float> [[SH]]
  %l = load float, ptr %p, align 4
  %sp = insertelement <4 x float> poison, float %l, i64 0
  %sh = shufflevector <4 x float> %sp, <4 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  ret <4 x float> %sh
}

; Ensure the loading of a small vector doesn't get widened.

define <4 x float> @load_v1f32_v4f32(ptr dereferenceable(16) %p) {
; CHECK-LABEL: @load_v1f32_v4f32(
; CHECK-NEXT:  [[L:%.*]] = load <1 x float>, ptr [[P:%.*]], align 4
; CHECK-NEXT:  [[SH:%.*]] = shufflevector <1 x float> %l, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
; CHECK-NEXT:  ret <4 x float> [[SH]]
  %l = load <1 x float>, ptr %p, align 4
  %s = shufflevector <1 x float> %l, <1 x float> poison, <4 x i32> <i32 0, i32 poison, i32 poison, i32 poison>
  ret <4 x float> %s
}

define <4 x float> @load_v2f32_v4f32(ptr align 16 dereferenceable(16) %p) {
; CHECK-LABEL: @load_v2f32_v4f32(
; CHECK-NEXT:  [[L:%.*]] = load <2 x float>, ptr [[P:%.*]], align 16
; CHECK-NEXT:  [[SH:%.*]] = shufflevector <2 x float> [[L]], <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
; CHECK-NEXT:  ret <4 x float> [[SH]]
  %l = load <2 x float>, ptr %p, align 16
  %s = shufflevector <2 x float> %l, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  ret <4 x float> %s
}
