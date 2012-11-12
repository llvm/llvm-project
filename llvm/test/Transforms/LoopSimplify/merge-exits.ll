; RUN: opt < %s -loop-simplify -loop-rotate -instcombine -indvars -S -verify-loop-info -verify-dom-info > %t
; RUN: not grep sext %t
; RUN: grep "phi i64" %t | count 1

; Loopsimplify should be able to merge the two loop exits
; into one, so that loop rotate can rotate the loop, so
; that indvars can promote the induction variable to i64
; without needing casts.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n32:64"

define float @t(float* %pTmp1, float* %peakWeight, i32 %bandEdgeIndex) nounwind {
entry:
	%t0 = load float* %peakWeight, align 4		; <float> [#uses=1]
	br label %bb1

bb:		; preds = %bb2
	%t1 = sext i32 %hiPart.0 to i64		; <i64> [#uses=1]
	%t2 = getelementptr float* %pTmp1, i64 %t1		; <float*> [#uses=1]
	%t3 = load float* %t2, align 4		; <float> [#uses=1]
	%t4 = fadd float %t3, %distERBhi.0		; <float> [#uses=1]
	%t5 = add i32 %hiPart.0, 1		; <i32> [#uses=2]
	%t6 = sext i32 %t5 to i64		; <i64> [#uses=1]
	%t7 = getelementptr float* %peakWeight, i64 %t6		; <float*> [#uses=1]
	%t8 = load float* %t7, align 4		; <float> [#uses=1]
	%t9 = fadd float %t8, %peakCount.0		; <float> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%peakCount.0 = phi float [ %t0, %entry ], [ %t9, %bb ]		; <float> [#uses=2]
	%hiPart.0 = phi i32 [ 0, %entry ], [ %t5, %bb ]		; <i32> [#uses=3]
	%distERBhi.0 = phi float [ 0.000000e+00, %entry ], [ %t4, %bb ]		; <float> [#uses=3]
	%t10 = fcmp uge float %distERBhi.0, 2.500000e+00		; <i1> [#uses=1]
	br i1 %t10, label %bb3, label %bb2

bb2:		; preds = %bb1
	%t11 = add i32 %bandEdgeIndex, -1		; <i32> [#uses=1]
	%t12 = icmp sgt i32 %t11, %hiPart.0		; <i1> [#uses=1]
	br i1 %t12, label %bb, label %bb3

bb3:		; preds = %bb2, %bb1
	%t13 = fdiv float %peakCount.0, %distERBhi.0		; <float> [#uses=1]
	ret float %t13
}
