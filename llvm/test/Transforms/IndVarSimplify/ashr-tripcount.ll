; RUN: opt < %s -indvars -S | FileCheck %s

; Indvars should be able to eliminate all of the sign extensions
; inside the loop.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"
@pow_2_tab = external constant [0 x float]		; <[0 x float]*> [#uses=1]
@pow_2_025_tab = external constant [0 x float]		; <[0 x float]*> [#uses=1]
@i_pow_2_tab = external constant [0 x float]		; <[0 x float]*> [#uses=1]
@i_pow_2_025_tab = external constant [0 x float]		; <[0 x float]*> [#uses=1]

define void @foo(i32 %gain, i32 %noOfLines, i32* %quaSpectrum, float* %iquaSpectrum, float* %pow4_3_tab_ptr) nounwind {
; CHECK-LABEL: @foo(
; CHECK: sext
; CHECK-NOT: sext
entry:
	%t0 = icmp slt i32 %gain, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb1, label %bb2

bb1:		; preds = %entry
	%t1 = sub i32 0, %gain		; <i32> [#uses=1]
	%t2 = sub i32 0, %gain		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1, %entry
	%pow_2_tab.pn = phi [0 x float]* [ @i_pow_2_tab, %bb1 ], [ @pow_2_tab, %entry ]		; <[0 x float]*> [#uses=1]
	%.pn3.in.in = phi i32 [ %t1, %bb1 ], [ %gain, %entry ]		; <i32> [#uses=1]
	%pow_2_025_tab.pn = phi [0 x float]* [ @i_pow_2_025_tab, %bb1 ], [ @pow_2_025_tab, %entry ]		; <[0 x float]*> [#uses=1]
	%.pn2.in.in = phi i32 [ %t2, %bb1 ], [ %gain, %entry ]		; <i32> [#uses=1]
	%.pn3.in = ashr i32 %.pn3.in.in, 2		; <i32> [#uses=1]
	%.pn2.in = and i32 %.pn2.in.in, 3		; <i32> [#uses=1]
	%.pn3 = sext i32 %.pn3.in to i64		; <i64> [#uses=1]
	%.pn2 = zext i32 %.pn2.in to i64		; <i64> [#uses=1]
	%.pn.in = getelementptr [0 x float], [0 x float]* %pow_2_tab.pn, i64 0, i64 %.pn3		; <float*> [#uses=1]
	%.pn1.in = getelementptr [0 x float], [0 x float]* %pow_2_025_tab.pn, i64 0, i64 %.pn2		; <float*> [#uses=1]
	%.pn = load float, float* %.pn.in		; <float> [#uses=1]
	%.pn1 = load float, float* %.pn1.in		; <float> [#uses=1]
	%invQuantizer.0 = fmul float %.pn, %.pn1		; <float> [#uses=4]
	%t3 = ashr i32 %noOfLines, 2		; <i32> [#uses=1]
	%t4 = icmp sgt i32 %t3, 0		; <i1> [#uses=1]
	br i1 %t4, label %bb.nph, label %return

bb.nph:		; preds = %bb2
	%t5 = ashr i32 %noOfLines, 2		; <i32> [#uses=1]
	br label %bb3

bb3:		; preds = %bb4, %bb.nph
	%i.05 = phi i32 [ %t49, %bb4 ], [ 0, %bb.nph ]		; <i32> [#uses=9]
	%k.04 = phi i32 [ %t48, %bb4 ], [ 0, %bb.nph ]		; <i32> [#uses=1]
	%t6 = sext i32 %i.05 to i64		; <i64> [#uses=1]
	%t7 = getelementptr i32, i32* %quaSpectrum, i64 %t6		; <i32*> [#uses=1]
	%t8 = load i32, i32* %t7, align 4		; <i32> [#uses=1]
	%t9 = zext i32 %t8 to i64		; <i64> [#uses=1]
	%t10 = getelementptr float, float* %pow4_3_tab_ptr, i64 %t9		; <float*> [#uses=1]
	%t11 = load float, float* %t10, align 4		; <float> [#uses=1]
	%t12 = or i32 %i.05, 1		; <i32> [#uses=1]
	%t13 = sext i32 %t12 to i64		; <i64> [#uses=1]
	%t14 = getelementptr i32, i32* %quaSpectrum, i64 %t13		; <i32*> [#uses=1]
	%t15 = load i32, i32* %t14, align 4		; <i32> [#uses=1]
	%t16 = zext i32 %t15 to i64		; <i64> [#uses=1]
	%t17 = getelementptr float, float* %pow4_3_tab_ptr, i64 %t16		; <float*> [#uses=1]
	%t18 = load float, float* %t17, align 4		; <float> [#uses=1]
	%t19 = or i32 %i.05, 2		; <i32> [#uses=1]
	%t20 = sext i32 %t19 to i64		; <i64> [#uses=1]
	%t21 = getelementptr i32, i32* %quaSpectrum, i64 %t20		; <i32*> [#uses=1]
	%t22 = load i32, i32* %t21, align 4		; <i32> [#uses=1]
	%t23 = zext i32 %t22 to i64		; <i64> [#uses=1]
	%t24 = getelementptr float, float* %pow4_3_tab_ptr, i64 %t23		; <float*> [#uses=1]
	%t25 = load float, float* %t24, align 4		; <float> [#uses=1]
	%t26 = or i32 %i.05, 3		; <i32> [#uses=1]
	%t27 = sext i32 %t26 to i64		; <i64> [#uses=1]
	%t28 = getelementptr i32, i32* %quaSpectrum, i64 %t27		; <i32*> [#uses=1]
	%t29 = load i32, i32* %t28, align 4		; <i32> [#uses=1]
	%t30 = zext i32 %t29 to i64		; <i64> [#uses=1]
	%t31 = getelementptr float, float* %pow4_3_tab_ptr, i64 %t30		; <float*> [#uses=1]
	%t32 = load float, float* %t31, align 4		; <float> [#uses=1]
	%t33 = fmul float %t11, %invQuantizer.0		; <float> [#uses=1]
	%t34 = sext i32 %i.05 to i64		; <i64> [#uses=1]
	%t35 = getelementptr float, float* %iquaSpectrum, i64 %t34		; <float*> [#uses=1]
	store float %t33, float* %t35, align 4
	%t36 = or i32 %i.05, 1		; <i32> [#uses=1]
	%t37 = fmul float %t18, %invQuantizer.0		; <float> [#uses=1]
	%t38 = sext i32 %t36 to i64		; <i64> [#uses=1]
	%t39 = getelementptr float, float* %iquaSpectrum, i64 %t38		; <float*> [#uses=1]
	store float %t37, float* %t39, align 4
	%t40 = or i32 %i.05, 2		; <i32> [#uses=1]
	%t41 = fmul float %t25, %invQuantizer.0		; <float> [#uses=1]
	%t42 = sext i32 %t40 to i64		; <i64> [#uses=1]
	%t43 = getelementptr float, float* %iquaSpectrum, i64 %t42		; <float*> [#uses=1]
	store float %t41, float* %t43, align 4
	%t44 = or i32 %i.05, 3		; <i32> [#uses=1]
	%t45 = fmul float %t32, %invQuantizer.0		; <float> [#uses=1]
	%t46 = sext i32 %t44 to i64		; <i64> [#uses=1]
	%t47 = getelementptr float, float* %iquaSpectrum, i64 %t46		; <float*> [#uses=1]
	store float %t45, float* %t47, align 4
	%t48 = add i32 %k.04, 1		; <i32> [#uses=2]
	%t49 = add i32 %i.05, 4		; <i32> [#uses=1]
	br label %bb4

bb4:		; preds = %bb3
	%t50 = icmp sgt i32 %t5, %t48		; <i1> [#uses=1]
	br i1 %t50, label %bb3, label %bb4.return_crit_edge

bb4.return_crit_edge:		; preds = %bb4
	br label %return

return:		; preds = %bb4.return_crit_edge, %bb2
	ret void
}
