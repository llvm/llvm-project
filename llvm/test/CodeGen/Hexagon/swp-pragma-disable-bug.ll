; RUN: llc -O2 -march=hexagon -enable-pipeliner  \
; RUN: -debug-only=pipeliner < %s 2>&1 > /dev/null | FileCheck %s
; REQUIRES: asserts
;
; Test that the pragma check that disables pipeliner does not disable pipelining
; on both the loops.
; CHECK: Can not pipeline loop
; CHECK-NOT: Can not pipeline loop

; Function Attrs: nofree norecurse nounwind
define dso_local i32 @foo(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture %c) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.023 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.023
  %0 = load i32, ptr %arrayidx, align 4, !tbaa !2
  %arrayidx1 = getelementptr inbounds i32, ptr %a, i32 %i.023
  %1 = load i32, ptr %arrayidx1, align 4, !tbaa !2
  %add = add nsw i32 %1, %0
  store i32 %add, ptr %arrayidx1, align 4, !tbaa !2
  %inc = add nuw nsw i32 %i.023, 1
  %exitcond24 = icmp eq i32 %inc, 10
  br i1 %exitcond24, label %for.body6, label %for.body, !llvm.loop !6

for.cond.cleanup5:                                ; preds = %for.body6
  ret i32 0

for.body6:                                        ; preds = %for.body, %for.body6
  %i2.022 = phi i32 [ %inc11, %for.body6 ], [ 0, %for.body ]
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i32 %i2.022
  %2 = load i32, ptr %arrayidx7, align 4, !tbaa !2
  %arrayidx8 = getelementptr inbounds i32, ptr %c, i32 %i2.022
  %3 = load i32, ptr %arrayidx8, align 4, !tbaa !2
  %add9 = add nsw i32 %3, %2
  store i32 %add9, ptr %arrayidx8, align 4, !tbaa !2
  %inc11 = add nuw nsw i32 %i2.022, 1
  %exitcond = icmp eq i32 %inc11, 10
  br i1 %exitcond, label %for.cond.cleanup5, label %for.body6, !llvm.loop !8
}

attributes #0 = { nofree norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+v60,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 6e29846b29d2bcaa8a7a3d869f24f242bd93d272)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.unroll.disable"}
!8 = distinct !{!8, !7, !9}
!9 = !{!"llvm.loop.pipeline.disable", i1 true}
