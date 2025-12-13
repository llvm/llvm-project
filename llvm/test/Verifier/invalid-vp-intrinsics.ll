; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare <4 x i32> @llvm.vp.fptosi.v4i32.v8f32(<8 x float>, <4 x i1>, i32)
declare <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float>, <4 x float>, metadata, <4 x i1>, i32)
declare <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32>, <4 x i32>, metadata, <4 x i1>, i32)

; CHECK: VP cast intrinsic first argument and result vector lengths must be equal
; CHECK-NEXT: %r0 = call <4 x i32>

define void @test_vp_fptosi(<8 x float> %src, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i32> @llvm.vp.fptosi.v4i32.v8f32(<8 x float> %src, <4 x i1> %m, i32 %n)
  ret void
}

; CHECK: invalid predicate for VP FP comparison intrinsic
; CHECK-NEXT: %r0 = call <4 x i1> @llvm.vp.fcmp.v4f32
; CHECK: invalid predicate for VP FP comparison intrinsic
; CHECK-NEXT: %r1 = call <4 x i1> @llvm.vp.fcmp.v4f32

define void @test_vp_fcmp(<4 x float> %a, <4 x float> %b, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float> %a, <4 x float> %b, metadata !"bad", <4 x i1> %m, i32 %n)
  %r1 = call <4 x i1> @llvm.vp.fcmp.v4f32(<4 x float> %a, <4 x float> %b, metadata !"eq", <4 x i1> %m, i32 %n)
  ret void
}

; CHECK: invalid predicate for VP integer comparison intrinsic
; CHECK-NEXT: %r0 = call <4 x i1> @llvm.vp.icmp.v4i32
; CHECK: invalid predicate for VP integer comparison intrinsic
; CHECK-NEXT: %r1 = call <4 x i1> @llvm.vp.icmp.v4i32

define void @test_vp_icmp(<4 x i32> %a, <4 x i32> %b, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> %a, <4 x i32> %b, metadata !"bad", <4 x i1> %m, i32 %n)
  %r1 = call <4 x i1> @llvm.vp.icmp.v4i32(<4 x i32> %a, <4 x i32> %b, metadata !"oeq", <4 x i1> %m, i32 %n)
  ret void
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx_neg3(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 -3, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <vscale x 2 x double> @splice_nxv2f64_idx_neg3_vscale_min1(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <vscale x 2 x double> @llvm.experimental.vp.splice.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 -3, <vscale x 2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <vscale x 2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <vscale x 2 x double> @splice_nxv2f64_idx_neg5_vscale_min2(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 %evl1, i32 %evl2) #1 {
  %res = call <vscale x 2 x double> @llvm.experimental.vp.splice.nxv2f64(<vscale x 2 x double> %a, <vscale x 2 x double> %b, i32 -5, <vscale x 2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <vscale x 2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx2(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #0 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 2, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

; CHECK: The splice index exceeds the range [-VL, VL-1] where VL is the known minimum number of elements in the vector
define <2 x double> @splice_v2f64_idx3(<2 x double> %a, <2 x double> %b, i32 %evl1, i32 %evl2) #1 {
  %res = call <2 x double> @llvm.experimental.vp.splice.v2f64(<2 x double> %a, <2 x double> %b, i32 4, <2 x i1> splat (i1 1), i32 %evl1, i32 %evl2)
  ret <2 x double> %res
}

attributes #0 = { vscale_range(1,16) }
attributes #1 = { vscale_range(2,16) }
