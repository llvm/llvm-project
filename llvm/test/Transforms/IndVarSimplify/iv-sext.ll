; RUN: opt < %s -passes=indvars -S | FileCheck %s

; Indvars should be able to promote the hiPart induction variable in the
; inner loop to i64.
; TODO: it should promote hiPart to i64 in the outer loop too.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"

define void @t(float* %pTmp1, float* %peakWeight, float* %nrgReducePeakrate, i32 %bandEdgeIndex, float %tmp1) nounwind {
entry:
	%tmp = load float, float* %peakWeight, align 4		; <float> [#uses=1]
	%tmp2 = icmp sgt i32 %bandEdgeIndex, 0		; <i1> [#uses=1]
	br i1 %tmp2, label %bb.nph22, label %return

bb.nph22:		; preds = %entry
	%tmp3 = add i32 %bandEdgeIndex, -1		; <i32> [#uses=2]
	br label %bb

; CHECK: bb:
; CHECK: phi i64
; CHECK-NOT: phi i64
bb:		; preds = %bb8, %bb.nph22
	%distERBhi.121 = phi float [ %distERBhi.2.lcssa, %bb8 ], [ 0.000000e+00, %bb.nph22 ]		; <float> [#uses=2]
	%distERBlo.120 = phi float [ %distERBlo.0.lcssa, %bb8 ], [ 0.000000e+00, %bb.nph22 ]		; <float> [#uses=2]
	%hiPart.119 = phi i32 [ %hiPart.0.lcssa, %bb8 ], [ 0, %bb.nph22 ]		; <i32> [#uses=3]
	%loPart.118 = phi i32 [ %loPart.0.lcssa, %bb8 ], [ 0, %bb.nph22 ]		; <i32> [#uses=2]
	%peakCount.117 = phi float [ %peakCount.2.lcssa, %bb8 ], [ %tmp, %bb.nph22 ]		; <float> [#uses=2]
	%part.016 = phi i32 [ %tmp46, %bb8 ], [ 0, %bb.nph22 ]		; <i32> [#uses=5]
	%tmp4 = icmp sgt i32 %part.016, 0		; <i1> [#uses=1]
	br i1 %tmp4, label %bb1, label %bb3.preheader

; CHECK: bb1:
bb1:		; preds = %bb
	%tmp5 = add i32 %part.016, -1		; <i32> [#uses=1]
	%tmp6 = sext i32 %tmp5 to i64		; <i64> [#uses=1]
	%tmp7 = getelementptr float, float* %pTmp1, i64 %tmp6		; <float*> [#uses=1]
	%tmp8 = load float, float* %tmp7, align 4		; <float> [#uses=1]
	%tmp9 = fadd float %tmp8, %distERBlo.120		; <float> [#uses=1]
	%tmp10 = add i32 %part.016, -1		; <i32> [#uses=1]
	%tmp11 = sext i32 %tmp10 to i64		; <i64> [#uses=1]
	%tmp12 = getelementptr float, float* %pTmp1, i64 %tmp11		; <float*> [#uses=1]
	%tmp13 = load float, float* %tmp12, align 4		; <float> [#uses=1]
	%tmp14 = fsub float %distERBhi.121, %tmp13		; <float> [#uses=1]
	br label %bb3.preheader

bb3.preheader:		; preds = %bb1, %bb
	%distERBlo.0.ph = phi float [ %distERBlo.120, %bb ], [ %tmp9, %bb1 ]		; <float> [#uses=3]
	%distERBhi.0.ph = phi float [ %distERBhi.121, %bb ], [ %tmp14, %bb1 ]		; <float> [#uses=3]
	%tmp15 = fcmp ogt float %distERBlo.0.ph, 2.500000e+00		; <i1> [#uses=1]
	br i1 %tmp15, label %bb.nph, label %bb5.preheader

bb.nph:		; preds = %bb3.preheader
	br label %bb2

bb2:		; preds = %bb3, %bb.nph
	%distERBlo.03 = phi float [ %tmp19, %bb3 ], [ %distERBlo.0.ph, %bb.nph ]		; <float> [#uses=1]
	%loPart.02 = phi i32 [ %tmp24, %bb3 ], [ %loPart.118, %bb.nph ]		; <i32> [#uses=3]
	%peakCount.01 = phi float [ %tmp23, %bb3 ], [ %peakCount.117, %bb.nph ]		; <float> [#uses=1]
	%tmp16 = sext i32 %loPart.02 to i64		; <i64> [#uses=1]
	%tmp17 = getelementptr float, float* %pTmp1, i64 %tmp16		; <float*> [#uses=1]
	%tmp18 = load float, float* %tmp17, align 4		; <float> [#uses=1]
	%tmp19 = fsub float %distERBlo.03, %tmp18		; <float> [#uses=3]
	%tmp20 = sext i32 %loPart.02 to i64		; <i64> [#uses=1]
	%tmp21 = getelementptr float, float* %peakWeight, i64 %tmp20		; <float*> [#uses=1]
	%tmp22 = load float, float* %tmp21, align 4		; <float> [#uses=1]
	%tmp23 = fsub float %peakCount.01, %tmp22		; <float> [#uses=2]
	%tmp24 = add i32 %loPart.02, 1		; <i32> [#uses=2]
	br label %bb3

bb3:		; preds = %bb2
	%tmp25 = fcmp ogt float %tmp19, 2.500000e+00		; <i1> [#uses=1]
	br i1 %tmp25, label %bb2, label %bb3.bb5.preheader_crit_edge

bb3.bb5.preheader_crit_edge:		; preds = %bb3
	%tmp24.lcssa = phi i32 [ %tmp24, %bb3 ]		; <i32> [#uses=1]
	%tmp23.lcssa = phi float [ %tmp23, %bb3 ]		; <float> [#uses=1]
	%tmp19.lcssa = phi float [ %tmp19, %bb3 ]		; <float> [#uses=1]
	br label %bb5.preheader

bb5.preheader:		; preds = %bb3.bb5.preheader_crit_edge, %bb3.preheader
	%distERBlo.0.lcssa = phi float [ %tmp19.lcssa, %bb3.bb5.preheader_crit_edge ], [ %distERBlo.0.ph, %bb3.preheader ]		; <float> [#uses=2]
	%loPart.0.lcssa = phi i32 [ %tmp24.lcssa, %bb3.bb5.preheader_crit_edge ], [ %loPart.118, %bb3.preheader ]		; <i32> [#uses=1]
	%peakCount.0.lcssa = phi float [ %tmp23.lcssa, %bb3.bb5.preheader_crit_edge ], [ %peakCount.117, %bb3.preheader ]		; <float> [#uses=2]
	%.not10 = fcmp olt float %distERBhi.0.ph, 2.500000e+00		; <i1> [#uses=1]
	%tmp26 = icmp sgt i32 %tmp3, %hiPart.119		; <i1> [#uses=1]
	%or.cond11 = and i1 %tmp26, %.not10		; <i1> [#uses=1]
	br i1 %or.cond11, label %bb.nph12, label %bb7

bb.nph12:		; preds = %bb5.preheader
	br label %bb4
; CHECK: bb4:
; CHECK: phi i64
; CHECK-NOT: phi i64
; CHECK-NOT: sext
bb4:		; preds = %bb5, %bb.nph12
	%distERBhi.29 = phi float [ %tmp30, %bb5 ], [ %distERBhi.0.ph, %bb.nph12 ]		; <float> [#uses=1]
	%hiPart.08 = phi i32 [ %tmp31, %bb5 ], [ %hiPart.119, %bb.nph12 ]		; <i32> [#uses=2]
	%peakCount.27 = phi float [ %tmp35, %bb5 ], [ %peakCount.0.lcssa, %bb.nph12 ]		; <float> [#uses=1]
	%tmp27 = sext i32 %hiPart.08 to i64		; <i64> [#uses=1]
	%tmp28 = getelementptr float, float* %pTmp1, i64 %tmp27		; <float*> [#uses=1]
	%tmp29 = load float, float* %tmp28, align 4		; <float> [#uses=1]
	%tmp30 = fadd float %tmp29, %distERBhi.29		; <float> [#uses=3]
	%tmp31 = add i32 %hiPart.08, 1		; <i32> [#uses=4]
	%tmp32 = sext i32 %tmp31 to i64		; <i64> [#uses=1]
	%tmp33 = getelementptr float, float* %peakWeight, i64 %tmp32		; <float*> [#uses=1]
	%tmp34 = load float, float* %tmp33, align 4		; <float> [#uses=1]
	%tmp35 = fadd float %tmp34, %peakCount.27		; <float> [#uses=2]
	br label %bb5

; CHECK: bb5:
bb5:		; preds = %bb4
	%.not = fcmp olt float %tmp30, 2.500000e+00		; <i1> [#uses=1]
	%tmp36 = icmp sgt i32 %tmp3, %tmp31		; <i1> [#uses=1]
	%or.cond = and i1 %tmp36, %.not		; <i1> [#uses=1]
	br i1 %or.cond, label %bb4, label %bb5.bb7_crit_edge

bb5.bb7_crit_edge:		; preds = %bb5
	%tmp35.lcssa = phi float [ %tmp35, %bb5 ]		; <float> [#uses=1]
	%tmp31.lcssa = phi i32 [ %tmp31, %bb5 ]		; <i32> [#uses=1]
	%tmp30.lcssa = phi float [ %tmp30, %bb5 ]		; <float> [#uses=1]
	br label %bb7

bb7:		; preds = %bb5.bb7_crit_edge, %bb5.preheader
	%distERBhi.2.lcssa = phi float [ %tmp30.lcssa, %bb5.bb7_crit_edge ], [ %distERBhi.0.ph, %bb5.preheader ]		; <float> [#uses=2]
	%hiPart.0.lcssa = phi i32 [ %tmp31.lcssa, %bb5.bb7_crit_edge ], [ %hiPart.119, %bb5.preheader ]		; <i32> [#uses=1]
	%peakCount.2.lcssa = phi float [ %tmp35.lcssa, %bb5.bb7_crit_edge ], [ %peakCount.0.lcssa, %bb5.preheader ]		; <float> [#uses=2]
	%tmp37 = fadd float %distERBlo.0.lcssa, %distERBhi.2.lcssa		; <float> [#uses=1]
	%tmp38 = fdiv float %peakCount.2.lcssa, %tmp37		; <float> [#uses=1]
	%tmp39 = fmul float %tmp38, %tmp1		; <float> [#uses=2]
	%tmp40 = fmul float %tmp39, %tmp39		; <float> [#uses=2]
	%tmp41 = fmul float %tmp40, %tmp40		; <float> [#uses=1]
	%tmp42 = fadd float %tmp41, 1.000000e+00		; <float> [#uses=1]
	%tmp43 = fdiv float 1.000000e+00, %tmp42		; <float> [#uses=1]
	%tmp44 = sext i32 %part.016 to i64		; <i64> [#uses=1]
	%tmp45 = getelementptr float, float* %nrgReducePeakrate, i64 %tmp44		; <float*> [#uses=1]
	store float %tmp43, float* %tmp45, align 4
	%tmp46 = add i32 %part.016, 1		; <i32> [#uses=2]
	br label %bb8

bb8:		; preds = %bb7
	%tmp47 = icmp slt i32 %tmp46, %bandEdgeIndex		; <i1> [#uses=1]
	br i1 %tmp47, label %bb, label %bb8.return_crit_edge

bb8.return_crit_edge:		; preds = %bb8
	br label %return

return:		; preds = %bb8.return_crit_edge, %entry
	ret void
}
