; RUN: opt -S -mtriple=aarch64 -loop-vectorize -force-vector-width=2 < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

@b = common local_unnamed_addr global i32 0, align 4
@a = common local_unnamed_addr global i16* null, align 8

define i32 @fn1() local_unnamed_addr #0 {
; We expect the backend to expand all reductions.
; CHECK: @llvm.vector.reduce
entry:
  %0 = load i32, i32* @b, align 4, !tbaa !1
  %cmp40 = icmp sgt i32 %0, 0
  br i1 %cmp40, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %1 = load i16*, i16** @a, align 8, !tbaa !5
  %2 = load i32, i32* @b, align 4, !tbaa !1
  %3 = sext i32 %2 to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %d.043 = phi i16 [ undef, %for.body.lr.ph ], [ %.sink28, %for.body ]
  %c.042 = phi i16 [ undef, %for.body.lr.ph ], [ %c.0., %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %1, i64 %indvars.iv
  %4 = load i16, i16* %arrayidx, align 2, !tbaa !7
  %cmp2 = icmp sgt i16 %c.042, %4
  %c.0. = select i1 %cmp2, i16 %c.042, i16 %4
  %cmp13 = icmp slt i16 %d.043, %4
  %.sink28 = select i1 %cmp13, i16 %d.043, i16 %4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv.next, %3
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  %c.0.lcssa = phi i16 [ undef, %entry ], [ %c.0., %for.body ]
  %d.0.lcssa = phi i16 [ undef, %entry ], [ %.sink28, %for.body ]
  %cmp26 = icmp sgt i16 %c.0.lcssa, %d.0.lcssa
  %conv27 = zext i1 %cmp26 to i32
  ret i32 %conv27
}

attributes #0 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }
!llvm.ident = !{!0}

!0 = !{!"clang"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !3, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !3, i64 0}
