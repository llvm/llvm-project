; RUN: opt -passes="print<cost-model>" -cost-kind=throughput 2>&1 -disable-output -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 < %s | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define <2 x float> @f32x2_add(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: 'f32x2_add'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %r = fadd <2 x float> %a, %b
  %r = fadd <2 x float> %a, %b
  ret <2 x float> %r
}

define <2 x float> @f32x2_sub(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: 'f32x2_sub'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %r = fsub <2 x float> %a, %b
  %r = fsub <2 x float> %a, %b
  ret <2 x float> %r
}

define <2 x float> @f32x2_mul(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: 'f32x2_mul'
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %r = fmul <2 x float> %a, %b
  %r = fmul <2 x float> %a, %b
  ret <2 x float> %r
}

define void @f32x2_extract(<2 x float> %v, ptr %out) {
; CHECK-LABEL: 'f32x2_extract'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %x0 = extractelement <2 x float> %v, i32 0
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %x1 = extractelement <2 x float> %v, i32 1
  %x0 = extractelement <2 x float> %v, i32 0
  %x1 = extractelement <2 x float> %v, i32 1
  store float %x0, ptr %out, align 4
  %p1 = getelementptr float, ptr %out, i64 1
  store float %x1, ptr %p1, align 4
  ret void
}

define void @f32x2_load_store(ptr %out, ptr %a, ptr %b) {
; CHECK-LABEL: 'f32x2_load_store'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %a0 = load <2 x float>, ptr %a, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %b0 = load <2 x float>, ptr %b, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %r = fadd <2 x float> %a0, %b0
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: store <2 x float> %r, ptr %out, align 8
  %a0 = load <2 x float>, ptr %a, align 8
  %b0 = load <2 x float>, ptr %b, align 8
  %r = fadd <2 x float> %a0, %b0
  store <2 x float> %r, ptr %out, align 8
  ret void
}

define <2 x float> @f32x2_splat(float %x) {
; CHECK-LABEL: 'f32x2_splat'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %v = insertelement <2 x float> poison, float %x, i32 0
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction: %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> zeroinitializer
  %v = insertelement <2 x float> poison, float %x, i32 0
  %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> zeroinitializer
  ret <2 x float> %s
}

define <2 x float> @f32x2_build(float %x, float %y) {
; CHECK-LABEL: 'f32x2_build'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %v0 = insertelement <2 x float> poison, float %x, i32 0
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %v1 = insertelement <2 x float> %v0, float %y, i32 1
  %v0 = insertelement <2 x float> poison, float %x, i32 0
  %v1 = insertelement <2 x float> %v0, float %y, i32 1
  ret <2 x float> %v1
}
