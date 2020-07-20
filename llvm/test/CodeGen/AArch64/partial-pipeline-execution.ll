; RUN: llc -O3 %s -o %t.s
; RUN: llc -O3 -stop-after=atomic-expand %s -o %t.mir
; RUN: llc -O3 -start-after=atomic-expand %s -o %t2.s

; If we add tti pass correctly files should be identical
; Otherwise LSR will use default TargetTransformInfo and
; optimize the loop differently
; RUN: cmp %t.s %t2.s

; ModuleID = 'loop.c'
source_filename = "loop.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-linux-gnu"

@q = dso_local local_unnamed_addr global i32* null, align 8

; Function Attrs: nofree norecurse nounwind
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #0 {
entry:
  %cmp5 = icmp sgt i32 %argc, 0
  br i1 %cmp5, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = load i32*, i32** @q, align 8, !tbaa !2
  %1 = zext i32 %argc to i64
  %2 = add nsw i64 %1, -1
  %3 = lshr i64 %2, 5
  %4 = add nuw nsw i64 %3, 1
  %min.iters.check = icmp eq i64 %3, 0
  br i1 %min.iters.check, label %for.body.preheader, label %vector.ph

for.body.preheader:                               ; preds = %middle.block, %for.body.lr.ph
  %indvars.iv.ph = phi i64 [ 0, %for.body.lr.ph ], [ %ind.end, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i64 %4, 1152921504606846974
  %ind.end = shl i64 %n.vec, 5
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %offset.idx = shl i64 %index, 5
  %induction7 = or i64 %offset.idx, 32
  %5 = getelementptr inbounds i32, i32* %0, i64 %offset.idx
  %6 = getelementptr inbounds i32, i32* %0, i64 %induction7
  %7 = trunc i64 %offset.idx to i32
  %8 = trunc i64 %induction7 to i32
  store i32 %7, i32* %5, align 4, !tbaa !6
  store i32 %8, i32* %6, align 4, !tbaa !6
  %index.next = add i64 %index, 2
  %9 = icmp eq i64 %index.next, %n.vec
  br i1 %9, label %middle.block, label %vector.body, !llvm.loop !8

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %4, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret i32 0

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ %indvars.iv.ph, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %0, i64 %indvars.iv
  %10 = trunc i64 %indvars.iv to i32
  store i32 %10, i32* %arrayidx, align 4, !tbaa !6
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 32
  %cmp = icmp ult i64 %indvars.iv.next, %1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !10
}

attributes #0 = { nofree norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git d9943e7f0ce888733ee7ba91da432e5f01f7aa85)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.isvectorized", i32 1}
!10 = distinct !{!10, !9}
