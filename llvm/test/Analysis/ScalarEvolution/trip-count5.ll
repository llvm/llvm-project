; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; ScalarEvolution should be able to compute a maximum trip count
; value sufficient to fold away both sext casts.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define float @t(float* %pTmp1, float* %peakWeight, float* %nrgReducePeakrate, i32 %bim) nounwind {
; CHECK-LABEL: Classifying expressions for: @t
entry:
	%tmp3 = load float, float* %peakWeight, align 4
	%tmp2538 = icmp sgt i32 %bim, 0
	br i1 %tmp2538, label %bb.nph, label %bb4

bb.nph:
	br label %bb

bb:
	%distERBhi.036 = phi float [ %tmp10, %bb1 ], [ 0.000000e+00, %bb.nph ]
	%hiPart.035 = phi i32 [ %tmp12, %bb1 ], [ 0, %bb.nph ]
	%peakCount.034 = phi float [ %tmp19, %bb1 ], [ %tmp3, %bb.nph ]
	%tmp6 = sext i32 %hiPart.035 to i64
	%tmp7 = getelementptr float, float* %pTmp1, i64 %tmp6
; CHECK:  %tmp6 = sext i32 %hiPart.035 to i64
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%bb>
	%tmp8 = load float, float* %tmp7, align 4
	%tmp10 = fadd float %tmp8, %distERBhi.036
	%tmp12 = add i32 %hiPart.035, 1
	%tmp15 = sext i32 %tmp12 to i64
	%tmp16 = getelementptr float, float* %peakWeight, i64 %tmp15
; CHECK:  %tmp15 = sext i32 %tmp12 to i64
; CHECK-NEXT:  -->  {1,+,1}<nuw><nsw><%bb>
	%tmp17 = load float, float* %tmp16, align 4
	%tmp19 = fadd float %tmp17, %peakCount.034
	br label %bb1

bb1:
	%tmp21 = fcmp olt float %tmp10, 2.500000e+00
	%tmp25 = icmp slt i32 %tmp12, %bim
	%tmp27 = and i1 %tmp21, %tmp25
	br i1 %tmp27, label %bb, label %bb1.bb4_crit_edge

bb1.bb4_crit_edge:
	br label %bb4

bb4:
	%distERBhi.0.lcssa = phi float [ %tmp10, %bb1.bb4_crit_edge ], [ 0.000000e+00, %entry ]
	%peakCount.0.lcssa = phi float [ %tmp19, %bb1.bb4_crit_edge ], [ %tmp3, %entry ]
	%tmp31 = fdiv float %peakCount.0.lcssa, %distERBhi.0.lcssa
	ret float %tmp31
}
