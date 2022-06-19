; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; Fold zeroing of inactive lanes into the load's passthrough parameter.
define <4 x float> @masked_load_and_zero_inactive_1(<4 x float>* %ptr, <4 x i1> %mask) {
; CHECK-LABEL: @masked_load_and_zero_inactive_1(
; CHECK: %load = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %ptr, i32 4, <4 x i1> %mask, <4 x float> zeroinitializer)
; CHECK-NEXT: ret <4 x float> %load
  %load = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %ptr, i32 4, <4 x i1> %mask, <4 x float> undef)
  %masked = select <4 x i1> %mask, <4 x float> %load, <4 x float> zeroinitializer
  ret <4 x float> %masked
}

; As above but reuse the load's existing passthrough.
define <4 x i32> @masked_load_and_zero_inactive_2(<4 x i32>* %ptr, <4 x i1> %mask) {
; CHECK-LABEL: @masked_load_and_zero_inactive_2(
; CHECK: %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
; CHECK-NEXT: ret <4 x i32> %load
  %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
  %masked = select <4 x i1> %mask, <4 x i32> %load, <4 x i32> zeroinitializer
  ret <4 x i32> %masked
}

; No transform when the load's passthrough cannot be reused or altered.
define <4 x i32> @masked_load_and_zero_inactive_3(<4 x i32>* %ptr, <4 x i1> %mask, <4 x i32> %passthrough) {
; CHECK-LABEL: @masked_load_and_zero_inactive_3(
; CHECK: %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> %passthrough)
; CHECK-NEXT: %masked = select <4 x i1> %mask, <4 x i32> %load, <4 x i32> zeroinitializer
; CHECK-NEXT: ret <4 x i32> %masked
  %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> %passthrough)
  %masked = select <4 x i1> %mask, <4 x i32> %load, <4 x i32> zeroinitializer
  ret <4 x i32> %masked
}

; Remove redundant select when its mask doesn't overlap with the load mask.
define <4 x i32> @masked_load_and_zero_inactive_4(<4 x i32>* %ptr, <4 x i1> %inv_mask) {
; CHECK-LABEL: @masked_load_and_zero_inactive_4(
; CHECK: %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT: %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
; CHECK-NEXT: ret <4 x i32> %load
  %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
  %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> undef)
  %masked = select <4 x i1> %inv_mask, <4 x i32> zeroinitializer, <4 x i32> %load
  ret <4 x i32> %masked
}

; As above but reuse the load's existing passthrough.
define <4 x i32> @masked_load_and_zero_inactive_5(<4 x i32>* %ptr, <4 x i1> %inv_mask) {
; CHECK-LABEL: @masked_load_and_zero_inactive_5(
; CHECK: %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT: %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
; CHECK-NEXT: ret <4 x i32> %load
  %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
  %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> zeroinitializer)
  %masked = select <4 x i1> %inv_mask, <4 x i32> zeroinitializer, <4 x i32> %load
  ret <4 x i32> %masked
}

; No transform when the load's passthrough cannot be reused or altered.
define <4 x i32> @masked_load_and_zero_inactive_6(<4 x i32>* %ptr, <4 x i1> %inv_mask, <4 x i32> %passthrough) {
; CHECK-LABEL: @masked_load_and_zero_inactive_6(
; CHECK: %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT: %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> %passthrough)
; CHECK-NEXT: %masked = select <4 x i1> %inv_mask, <4 x i32> zeroinitializer, <4 x i32> %load
; CHECK-NEXT: ret <4 x i32> %masked
  %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
  %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask, <4 x i32> %passthrough)
  %masked = select <4 x i1> %inv_mask, <4 x i32> zeroinitializer, <4 x i32> %load
  ret <4 x i32> %masked
}

; No transform when select and load masks have no relation.
define <4 x i32> @masked_load_and_zero_inactive_7(<4 x i32>* %ptr, <4 x i1> %mask1, <4 x i1> %mask2) {
; CHECK-LABEL: @masked_load_and_zero_inactive_7(
; CHECK: %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask1, <4 x i32> zeroinitializer)
; CHECK-NEXT: %masked = select <4 x i1> %mask2, <4 x i32> zeroinitializer, <4 x i32> %load
; CHECK-NEXT: ret <4 x i32> %masked
  %load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %ptr, i32 4, <4 x i1> %mask1, <4 x i32> zeroinitializer)
  %masked = select <4 x i1> %mask2, <4 x i32> zeroinitializer, <4 x i32> %load
  ret <4 x i32> %masked
}

; A more complex case where we can prove the select mask is a subset of the
; load's inactive lanes and thus the load's passthrough takes effect.
define <4 x float> @masked_load_and_zero_inactive_8(<4 x float>* %ptr, <4 x i1> %inv_mask, <4 x i1> %cond) {
; CHECK-LABEL: @masked_load_and_zero_inactive_8(
; CHECK: %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT: %pg = and <4 x i1> %mask, %cond
; CHECK-NEXT: %load = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %ptr, i32 4, <4 x i1> %pg, <4 x float> zeroinitializer)
; CHECK-NEXT: ret <4 x float> %load
  %mask = xor <4 x i1> %inv_mask, <i1 true, i1 true, i1 true, i1 true>
  %pg = and <4 x i1> %mask, %cond
  %load = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %ptr, i32 4, <4 x i1> %pg, <4 x float> undef)
  %masked = select <4 x i1> %inv_mask, <4 x float> zeroinitializer, <4 x float> %load
  ret <4 x float> %masked
}

define <8 x float> @masked_load_and_scalar_select_cond(<8 x float>* %ptr, <8 x i1> %mask, i1 %cond) {
; CHECK-LABEL: @masked_load_and_scalar_select_cond(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call <8 x float> @llvm.masked.load.v8f32.p0v8f32(<8 x float>* [[PTR:%.*]], i32 32, <8 x i1> [[MASK:%.*]], <8 x float> undef)
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[COND:%.*]], <8 x float> zeroinitializer, <8 x float> [[TMP0]]
; CHECK-NEXT:    ret <8 x float> [[TMP1]]
entry:
  %0 = call <8 x float> @llvm.masked.load.v8f32.p0v8f32(<8 x float>* %ptr, i32 32, <8 x i1> %mask, <8 x float> undef)
  %1 = select i1 %cond, <8 x float> zeroinitializer, <8 x float> %0
  ret <8 x float> %1
}

declare <8 x float> @llvm.masked.load.v8f32.p0v8f32(<8 x float>*, i32 immarg, <8 x i1>, <8 x float>)
declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>*, i32 immarg, <4 x i1>, <4 x float>)
