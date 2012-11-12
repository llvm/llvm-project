; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s

@in = global float 0x400921FA00000000, align 4

; Test signed conversion.
; CHECK: t0
; CHECK-NOT: vmul
define void @t0() nounwind {
entry:
  %tmp = load float* @in, align 4, !tbaa !0
  %vecinit.i = insertelement <2 x float> undef, float %tmp, i32 0
  %vecinit2.i = insertelement <2 x float> %vecinit.i, float %tmp, i32 1
  %mul.i = fmul <2 x float> %vecinit2.i, <float 8.000000e+00, float 8.000000e+00>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  tail call void @foo_int32x2_t(<2 x i32> %vcvt.i) nounwind
  ret void
}

declare void @foo_int32x2_t(<2 x i32>)

; Test unsigned conversion.
; CHECK: t1
; CHECK-NOT: vmul
define void @t1() nounwind {
entry:
  %tmp = load float* @in, align 4, !tbaa !0
  %vecinit.i = insertelement <2 x float> undef, float %tmp, i32 0
  %vecinit2.i = insertelement <2 x float> %vecinit.i, float %tmp, i32 1
  %mul.i = fmul <2 x float> %vecinit2.i, <float 8.000000e+00, float 8.000000e+00>
  %vcvt.i = fptoui <2 x float> %mul.i to <2 x i32>
  tail call void @foo_uint32x2_t(<2 x i32> %vcvt.i) nounwind
  ret void
}

declare void @foo_uint32x2_t(<2 x i32>)

; Test which should not fold due to non-power of 2.
; CHECK: t2
; CHECK: vmul
define void @t2() nounwind {
entry:
  %tmp = load float* @in, align 4, !tbaa !0
  %vecinit.i = insertelement <2 x float> undef, float %tmp, i32 0
  %vecinit2.i = insertelement <2 x float> %vecinit.i, float %tmp, i32 1
  %mul.i = fmul <2 x float> %vecinit2.i, <float 0x401B333340000000, float 0x401B333340000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  tail call void @foo_int32x2_t(<2 x i32> %vcvt.i) nounwind
  ret void
}

; Test which should not fold due to power of 2 out of range.
; CHECK: t3
; CHECK: vmul
define void @t3() nounwind {
entry:
  %tmp = load float* @in, align 4, !tbaa !0
  %vecinit.i = insertelement <2 x float> undef, float %tmp, i32 0
  %vecinit2.i = insertelement <2 x float> %vecinit.i, float %tmp, i32 1
  %mul.i = fmul <2 x float> %vecinit2.i, <float 0x4200000000000000, float 0x4200000000000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  tail call void @foo_int32x2_t(<2 x i32> %vcvt.i) nounwind
  ret void
}

; Test which case where const is max power of 2 (i.e., 2^32).
; CHECK: t4
; CHECK-NOT: vmul
define void @t4() nounwind {
entry:
  %tmp = load float* @in, align 4, !tbaa !0
  %vecinit.i = insertelement <2 x float> undef, float %tmp, i32 0
  %vecinit2.i = insertelement <2 x float> %vecinit.i, float %tmp, i32 1
  %mul.i = fmul <2 x float> %vecinit2.i, <float 0x41F0000000000000, float 0x41F0000000000000>
  %vcvt.i = fptosi <2 x float> %mul.i to <2 x i32>
  tail call void @foo_int32x2_t(<2 x i32> %vcvt.i) nounwind
  ret void
}

; Test quadword.
; CHECK: t5
; CHECK-NOT: vmul
define void @t5() nounwind {
entry:
  %tmp = load float* @in, align 4, !tbaa !0
  %vecinit.i = insertelement <4 x float> undef, float %tmp, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %tmp, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %tmp, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %tmp, i32 3
  %mul.i = fmul <4 x float> %vecinit6.i, <float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00>
  %vcvt.i = fptosi <4 x float> %mul.i to <4 x i32>
  tail call void @foo_int32x4_t(<4 x i32> %vcvt.i) nounwind
  ret void
}

declare void @foo_int32x4_t(<4 x i32>)

!0 = metadata !{metadata !"float", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
