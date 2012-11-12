; RUN: llc < %s -mtriple=armv7-apple-ios | FileCheck %s

@in = global float 0x400921FA00000000, align 4
@iin = global i32 -1023, align 4
@uin = global i32 1023, align 4

declare void @foo_int32x4_t(<4 x i32>)

; Test signed conversion.
; CHECK: t1
; CHECK-NOT: {{vdiv|vmul}}
define void @t1() nounwind {
entry:
  %tmp = load i32* @iin, align 4, !tbaa !3
  %vecinit.i = insertelement <2 x i32> undef, i32 %tmp, i32 0
  %vecinit2.i = insertelement <2 x i32> %vecinit.i, i32 %tmp, i32 1
  %vcvt.i = sitofp <2 x i32> %vecinit2.i to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 8.000000e+00, float 8.000000e+00>
  tail call void @foo_float32x2_t(<2 x float> %div.i) nounwind
  ret void
}

declare void @foo_float32x2_t(<2 x float>)

; Test unsigned conversion.
; CHECK: t2
; CHECK-NOT: {{vdiv|vmul}}
define void @t2() nounwind {
entry:
  %tmp = load i32* @uin, align 4, !tbaa !3
  %vecinit.i = insertelement <2 x i32> undef, i32 %tmp, i32 0
  %vecinit2.i = insertelement <2 x i32> %vecinit.i, i32 %tmp, i32 1
  %vcvt.i = uitofp <2 x i32> %vecinit2.i to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 8.000000e+00, float 8.000000e+00>
  tail call void @foo_float32x2_t(<2 x float> %div.i) nounwind
  ret void
}

; Test which should not fold due to non-power of 2.
; CHECK: t3
; CHECK: {{vdiv|vmul}}
define void @t3() nounwind {
entry:
  %tmp = load i32* @iin, align 4, !tbaa !3
  %vecinit.i = insertelement <2 x i32> undef, i32 %tmp, i32 0
  %vecinit2.i = insertelement <2 x i32> %vecinit.i, i32 %tmp, i32 1
  %vcvt.i = sitofp <2 x i32> %vecinit2.i to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 0x401B333340000000, float 0x401B333340000000>
  tail call void @foo_float32x2_t(<2 x float> %div.i) nounwind
  ret void
}

; Test which should not fold due to power of 2 out of range.
; CHECK: t4
; CHECK: {{vdiv|vmul}}
define void @t4() nounwind {
entry:
  %tmp = load i32* @iin, align 4, !tbaa !3
  %vecinit.i = insertelement <2 x i32> undef, i32 %tmp, i32 0
  %vecinit2.i = insertelement <2 x i32> %vecinit.i, i32 %tmp, i32 1
  %vcvt.i = sitofp <2 x i32> %vecinit2.i to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 0x4200000000000000, float 0x4200000000000000>
  tail call void @foo_float32x2_t(<2 x float> %div.i) nounwind
  ret void
}

; Test case where const is max power of 2 (i.e., 2^32).
; CHECK: t5
; CHECK-NOT: {{vdiv|vmul}}
define void @t5() nounwind {
entry:
  %tmp = load i32* @iin, align 4, !tbaa !3
  %vecinit.i = insertelement <2 x i32> undef, i32 %tmp, i32 0
  %vecinit2.i = insertelement <2 x i32> %vecinit.i, i32 %tmp, i32 1
  %vcvt.i = sitofp <2 x i32> %vecinit2.i to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 0x41F0000000000000, float 0x41F0000000000000>
  tail call void @foo_float32x2_t(<2 x float> %div.i) nounwind
  ret void
}

; Test quadword.
; CHECK: t6
; CHECK-NOT: {{vdiv|vmul}}
define void @t6() nounwind {
entry:
  %tmp = load i32* @iin, align 4, !tbaa !3
  %vecinit.i = insertelement <4 x i32> undef, i32 %tmp, i32 0
  %vecinit2.i = insertelement <4 x i32> %vecinit.i, i32 %tmp, i32 1
  %vecinit4.i = insertelement <4 x i32> %vecinit2.i, i32 %tmp, i32 2
  %vecinit6.i = insertelement <4 x i32> %vecinit4.i, i32 %tmp, i32 3
  %vcvt.i = sitofp <4 x i32> %vecinit6.i to <4 x float>
  %div.i = fdiv <4 x float> %vcvt.i, <float 8.000000e+00, float 8.000000e+00, float 8.000000e+00, float 8.000000e+00>
  tail call void @foo_float32x4_t(<4 x float> %div.i) nounwind
  ret void
}

declare void @foo_float32x4_t(<4 x float>)

!0 = metadata !{metadata !"float", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"int", metadata !1}
