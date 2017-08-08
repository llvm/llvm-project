; This test checks that the given loop still beneficial for vecotization
; even if it contains scalarized load (gather on AVX2)
;RUN: opt < %s -loop-vectorize -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readonly uwtable
define i32 @matrix_row_col([100 x i32]* nocapture readonly %data, i32 %i, i32 %j) local_unnamed_addr #0 {
entry:
  %idxprom = sext i32 %i to i64
  %idxprom5 = sext i32 %j to i64
  br label %for.body

  for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add7

  for.body:                                         ; preds = %for.body, %entry
  ; the loop gets vectorized
  ; first consecutive load as vector load
  ; CHECK: %wide.load = load <8 x i32>
  ; second strided load scalarized
  ; CHECK: load i32
  ; CHECK: load i32
  ; CHECK: load i32
  ; CHECK: load i32
  ; CHECK: load i32
  ; CHECK: load i32
  ; CHECK: load i32
  ; CHECK: load i32

  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.015 = phi i32 [ 0, %entry ], [ %add7, %for.body ]
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32]* %data, i64 %idxprom, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx2, align 4, !tbaa !1
  %arrayidx6 = getelementptr inbounds [100 x i32], [100 x i32]* %data, i64 %indvars.iv, i64 %idxprom5
  %1 = load i32, i32* %arrayidx6, align 4, !tbaa !1
  %mul = mul nsw i32 %1, %0
  %add = add i32 %sum.015, 4
  %add7 = add i32 %add, %mul
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { "target-cpu"="core-avx2" "target-features"="+avx,+avx2,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (cfe/trunk 284570)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
