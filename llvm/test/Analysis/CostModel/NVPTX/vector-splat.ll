; RUN: opt -passes="print<cost-model>" -cost-kind=throughput 2>&1 -disable-output -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 < %s | FileCheck %s --check-prefixes=CHECK,SM100
; RUN: opt -passes="print<cost-model>" -cost-kind=throughput 2>&1 -disable-output -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 < %s | FileCheck %s --check-prefix=SM90
; RUN: opt -passes="print<cost-model>" -cost-kind=code-size 2>&1 -disable-output -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 < %s | FileCheck %s --check-prefix=SIZE

target triple = "nvptx64-nvidia-cuda"

define <2 x float> @f32x2_splat(float %x) {
; CHECK-LABEL: 'f32x2_splat'
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction: %v = insertelement <2 x float> poison, float %x, i32 0
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction: %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> zeroinitializer
  %v = insertelement <2 x float> poison, float %x, i32 0
  %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> zeroinitializer
  ret <2 x float> %s
}

define <2 x float> @f32x2_splat_two_src(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: 'f32x2_splat_two_src'
; SM100-NEXT: Cost Model: Found an estimated cost of 0 for instruction: %s = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> zeroinitializer
; SM90-LABEL: 'f32x2_splat_two_src'
; SM90-NEXT: Cost Model: Found an estimated cost of 3 for instruction: %s = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> zeroinitializer
; SIZE-LABEL: 'f32x2_splat_two_src'
; SIZE-NEXT: Cost Model: Found an estimated cost of 3 for instruction: %s = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> zeroinitializer
  %s = shufflevector <2 x float> %a, <2 x float> %b, <2 x i32> zeroinitializer
  ret <2 x float> %s
}

define <2 x float> @f32x2_broadcast_lane1(<2 x float> %v) {
; CHECK-LABEL: 'f32x2_broadcast_lane1'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction: %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x float> %s
}

define <2 x float> @f32x2_reverse(<2 x float> %v) {
; CHECK-LABEL: 'f32x2_reverse'
; CHECK-NEXT: Cost Model: Found an estimated cost of 4 for instruction: %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  %s = shufflevector <2 x float> %v, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  ret <2 x float> %s
}

define <2 x half> @f16x2_splat(<2 x half> %v) {
; CHECK-LABEL: 'f16x2_splat'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction: %s = shufflevector <2 x half> %v, <2 x half> poison, <2 x i32> zeroinitializer
  %s = shufflevector <2 x half> %v, <2 x half> poison, <2 x i32> zeroinitializer
  ret <2 x half> %s
}

define <2 x i32> @i32x2_splat(<2 x i32> %v) {
; CHECK-LABEL: 'i32x2_splat'
; CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction: %s = shufflevector <2 x i32> %v, <2 x i32> poison, <2 x i32> zeroinitializer
  %s = shufflevector <2 x i32> %v, <2 x i32> poison, <2 x i32> zeroinitializer
  ret <2 x i32> %s
}

define <4 x float> @f32x4_splat(<4 x float> %v) {
; CHECK-LABEL: 'f32x4_splat'
; CHECK-NEXT: Cost Model: Found an estimated cost of 5 for instruction: %s = shufflevector <4 x float> %v, <4 x float> poison, <4 x i32> zeroinitializer
  %s = shufflevector <4 x float> %v, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %s
}
