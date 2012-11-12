; RUN: opt < %s -analyze -scalar-evolution > %t
; RUN: grep sext %t | count 2
; RUN: not grep "(sext" %t

; ScalarEvolution should be able to compute a maximum trip count
; value sufficient to fold away both sext casts.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define float @t(float* %pTmp1, float* %peakWeight, float* %nrgReducePeakrate, i32 %bim) nounwind {
entry:
	%tmp3 = load float* %peakWeight, align 4		; <float> [#uses=2]
	%tmp2538 = icmp sgt i32 %bim, 0		; <i1> [#uses=1]
	br i1 %tmp2538, label %bb.nph, label %bb4

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%distERBhi.036 = phi float [ %tmp10, %bb1 ], [ 0.000000e+00, %bb.nph ]		; <float> [#uses=1]
	%hiPart.035 = phi i32 [ %tmp12, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=2]
	%peakCount.034 = phi float [ %tmp19, %bb1 ], [ %tmp3, %bb.nph ]		; <float> [#uses=1]
	%tmp6 = sext i32 %hiPart.035 to i64		; <i64> [#uses=1]
	%tmp7 = getelementptr float* %pTmp1, i64 %tmp6		; <float*> [#uses=1]
	%tmp8 = load float* %tmp7, align 4		; <float> [#uses=1]
	%tmp10 = fadd float %tmp8, %distERBhi.036		; <float> [#uses=3]
	%tmp12 = add i32 %hiPart.035, 1		; <i32> [#uses=3]
	%tmp15 = sext i32 %tmp12 to i64		; <i64> [#uses=1]
	%tmp16 = getelementptr float* %peakWeight, i64 %tmp15		; <float*> [#uses=1]
	%tmp17 = load float* %tmp16, align 4		; <float> [#uses=1]
	%tmp19 = fadd float %tmp17, %peakCount.034		; <float> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%tmp21 = fcmp olt float %tmp10, 2.500000e+00		; <i1> [#uses=1]
	%tmp25 = icmp slt i32 %tmp12, %bim		; <i1> [#uses=1]
	%tmp27 = and i1 %tmp21, %tmp25		; <i1> [#uses=1]
	br i1 %tmp27, label %bb, label %bb1.bb4_crit_edge

bb1.bb4_crit_edge:		; preds = %bb1
	br label %bb4

bb4:		; preds = %bb1.bb4_crit_edge, %entry
	%distERBhi.0.lcssa = phi float [ %tmp10, %bb1.bb4_crit_edge ], [ 0.000000e+00, %entry ]		; <float> [#uses=1]
	%peakCount.0.lcssa = phi float [ %tmp19, %bb1.bb4_crit_edge ], [ %tmp3, %entry ]		; <float> [#uses=1]
	%tmp31 = fdiv float %peakCount.0.lcssa, %distERBhi.0.lcssa		; <float> [#uses=1]
	ret float %tmp31
}
