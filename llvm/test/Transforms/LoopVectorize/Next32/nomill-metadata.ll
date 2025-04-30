; RUN: opt < %s -mtriple=x86_64 -passes=loop-vectorize --no-mill-remainder-loop -S 2>&1 | FileCheck %s

; void foo(int *a, int size) {
;   for(int i = 0; i < size; ++i)
;     a[i] += 1;
; }

; CHECK-LABEL: @foo(
; CHECK: br i1 {{.*}}, label {{.*}}, label %vector.body, !llvm.loop ![[LOOP_7:.*]]
; CHECK: br i1 {{.*}}, label {{.*}}, label %for.body, !llvm.loop ![[LOOP_10:.*]]
define dso_local void @foo(i32* nocapture noundef %a, i32 noundef %size) local_unnamed_addr #0 {
entry:
  %cmp4 = icmp sgt i32 %size, 0
  br i1 %cmp4, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %idxprom = zext i32 %i.05 to i64
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4, !tbaa !3
  %add = add nsw i32 %0, 1
  store i32 %add, i32* %arrayidx, align 4, !tbaa !3
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond.not = icmp eq i32 %inc, %size
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !7
}

attributes #0 = { nofree norecurse nosync nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8"}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"NextSilicon clang version 14.0.6 (git@github.com:nextsilicon/next-llvm-project.git e1b89b08af985a2bcd11cac246f3b0c7c019382e)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}

; CHECK: ![[LOOP_7:.*]] = distinct !{![[LOOP_7:.*]], ![[MUST_PROGRESS_VEC:.*]], ![[IS_VECTORIZED_VEC:.*]]}
; CHECK: ![[MUST_PROGRESS_VEC:.*]] = !{!"llvm.loop.mustprogress"}
; CHECK: ![[IS_VECTORIZED_VEC:.*]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: ![[LOOP_10:.*]] = distinct !{![[LOOP_10:.*]], ![[MUST_PROGRESS:.*]], [[UNROLL_DISABLE:.*]], ![[NO_MILL_REMAINDER:.*]], ![[IS_VECTORIZED:.*]]}
; CHECK: ![[NO_MILL_REMAINDER:.*]] = !{!"ns.loop.mark", !"nomill"}
