; RUN: opt %loadNPMPolly -pass-remarks-analysis="polly-scops" '-passes=print<polly-function-scops>' \
; RUN:     -polly-invariant-load-hoisting=true \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s
;
; This test case has an SCEVSMax expression with a very high arity. The
; piecewise affine function we would create for it would have a huge amount of
; conjuncts, thus it would take a lot of time creating and handling it.
;
; This ensures we bail out for really complex expressions:
;
; CHECK: Low complexity assumption: {  : false }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; Function Attrs: norecurse nounwind
define i32 @foo(ptr nocapture readonly %src1, ptr nocapture readonly %src2, ptr nocapture %score, ptr nocapture %max, i32 %n) #0 {
entry:
  %cmp33 = icmp sgt i32 %n, 0
  br i1 %cmp33, label %for.body.preheader, label %for.body7.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body7.preheader.loopexit:                     ; preds = %for.body
  br label %for.body7.preheader

for.body7.preheader:                              ; preds = %for.body7.preheader.loopexit, %entry
  %0 = load i32, ptr %score, align 4, !tbaa !3
  %cmp9 = icmp sgt i32 %0, -1
  %.scoreMax.0 = select i1 %cmp9, i32 %0, i32 -1
  %arrayidx8.1 = getelementptr inbounds i32, ptr %score, i32 1
  %1 = load i32, ptr %arrayidx8.1, align 4, !tbaa !3
  %cmp9.1 = icmp sgt i32 %1, %.scoreMax.0
  %.scoreMax.0.1 = select i1 %cmp9.1, i32 %1, i32 %.scoreMax.0
  %arrayidx8.2 = getelementptr inbounds i32, ptr %score, i32 2
  %2 = load i32, ptr %arrayidx8.2, align 4, !tbaa !3
  %cmp9.2 = icmp sgt i32 %2, %.scoreMax.0.1
  %.scoreMax.0.2 = select i1 %cmp9.2, i32 %2, i32 %.scoreMax.0.1
  %arrayidx8.3 = getelementptr inbounds i32, ptr %score, i32 3
  %3 = load i32, ptr %arrayidx8.3, align 4, !tbaa !3
  %cmp9.3 = icmp sgt i32 %3, %.scoreMax.0.2
  %.scoreMax.0.3 = select i1 %cmp9.3, i32 %3, i32 %.scoreMax.0.2
  %arrayidx8.4 = getelementptr inbounds i32, ptr %score, i32 4
  %4 = load i32, ptr %arrayidx8.4, align 4, !tbaa !3
  %cmp9.4 = icmp sgt i32 %4, %.scoreMax.0.3
  %.scoreMax.0.4 = select i1 %cmp9.4, i32 %4, i32 %.scoreMax.0.3
  %arrayidx8.5 = getelementptr inbounds i32, ptr %score, i32 5
  %5 = load i32, ptr %arrayidx8.5, align 4, !tbaa !3
  %cmp9.5 = icmp sgt i32 %5, %.scoreMax.0.4
  %.scoreMax.0.5 = select i1 %cmp9.5, i32 %5, i32 %.scoreMax.0.4
  %arrayidx8.6 = getelementptr inbounds i32, ptr %score, i32 6
  %6 = load i32, ptr %arrayidx8.6, align 4, !tbaa !3
  %cmp9.6 = icmp sgt i32 %6, %.scoreMax.0.5
  %.scoreMax.0.6 = select i1 %cmp9.6, i32 %6, i32 %.scoreMax.0.5
  %arrayidx8.7 = getelementptr inbounds i32, ptr %score, i32 7
  %7 = load i32, ptr %arrayidx8.7, align 4, !tbaa !3
  %cmp9.7 = icmp sgt i32 %7, %.scoreMax.0.6
  %.scoreMax.0.7 = select i1 %cmp9.7, i32 %7, i32 %.scoreMax.0.6
  %arrayidx8.8 = getelementptr inbounds i32, ptr %score, i32 8
  %8 = load i32, ptr %arrayidx8.8, align 4, !tbaa !3
  %cmp9.8 = icmp sgt i32 %8, %.scoreMax.0.7
  %.scoreMax.0.8 = select i1 %cmp9.8, i32 %8, i32 %.scoreMax.0.7
  %arrayidx8.9 = getelementptr inbounds i32, ptr %score, i32 9
  %9 = load i32, ptr %arrayidx8.9, align 4, !tbaa !3
  %cmp9.9 = icmp sgt i32 %9, %.scoreMax.0.8
  %.scoreMax.0.9 = select i1 %cmp9.9, i32 %9, i32 %.scoreMax.0.8
  %arrayidx8.10 = getelementptr inbounds i32, ptr %score, i32 10
  %10 = load i32, ptr %arrayidx8.10, align 4, !tbaa !3
  %cmp9.10 = icmp sgt i32 %10, %.scoreMax.0.9
  %.scoreMax.0.10 = select i1 %cmp9.10, i32 %10, i32 %.scoreMax.0.9
  %arrayidx8.11 = getelementptr inbounds i32, ptr %score, i32 11
  %11 = load i32, ptr %arrayidx8.11, align 4, !tbaa !3
  %cmp9.11 = icmp sgt i32 %11, %.scoreMax.0.10
  %.scoreMax.0.11 = select i1 %cmp9.11, i32 %11, i32 %.scoreMax.0.10
  %arrayidx8.12 = getelementptr inbounds i32, ptr %score, i32 12
  %12 = load i32, ptr %arrayidx8.12, align 4, !tbaa !3
  %cmp9.12 = icmp sgt i32 %12, %.scoreMax.0.11
  %.scoreMax.0.12 = select i1 %cmp9.12, i32 %12, i32 %.scoreMax.0.11
  %arrayidx8.13 = getelementptr inbounds i32, ptr %score, i32 13
  %13 = load i32, ptr %arrayidx8.13, align 4, !tbaa !3
  %cmp9.13 = icmp sgt i32 %13, %.scoreMax.0.12
  %.scoreMax.0.13 = select i1 %cmp9.13, i32 %13, i32 %.scoreMax.0.12
  %arrayidx8.14 = getelementptr inbounds i32, ptr %score, i32 14
  %14 = load i32, ptr %arrayidx8.14, align 4, !tbaa !3
  %cmp9.14 = icmp sgt i32 %14, %.scoreMax.0.13
  %.scoreMax.0.14 = select i1 %cmp9.14, i32 %14, i32 %.scoreMax.0.13
  %arrayidx8.15 = getelementptr inbounds i32, ptr %score, i32 15
  %15 = load i32, ptr %arrayidx8.15, align 4, !tbaa !3
  %cmp9.15 = icmp sgt i32 %15, %.scoreMax.0.14
  %.scoreMax.0.15 = select i1 %cmp9.15, i32 %15, i32 %.scoreMax.0.14
  %arrayidx8.16 = getelementptr inbounds i32, ptr %score, i32 16
  %16 = load i32, ptr %arrayidx8.16, align 4, !tbaa !3
  %cmp9.16 = icmp sgt i32 %16, %.scoreMax.0.15
  %.scoreMax.0.16 = select i1 %cmp9.16, i32 %16, i32 %.scoreMax.0.15
  %arrayidx8.17 = getelementptr inbounds i32, ptr %score, i32 17
  %17 = load i32, ptr %arrayidx8.17, align 4, !tbaa !3
  %cmp9.17 = icmp sgt i32 %17, %.scoreMax.0.16
  %.scoreMax.0.17 = select i1 %cmp9.17, i32 %17, i32 %.scoreMax.0.16
  %arrayidx8.18 = getelementptr inbounds i32, ptr %score, i32 18
  %18 = load i32, ptr %arrayidx8.18, align 4, !tbaa !3
  %cmp9.18 = icmp sgt i32 %18, %.scoreMax.0.17
  %.scoreMax.0.18 = select i1 %cmp9.18, i32 %18, i32 %.scoreMax.0.17
  %arrayidx8.19 = getelementptr inbounds i32, ptr %score, i32 19
  %19 = load i32, ptr %arrayidx8.19, align 4, !tbaa !3
  %cmp9.19 = icmp sgt i32 %19, %.scoreMax.0.18
  %.scoreMax.0.19 = select i1 %cmp9.19, i32 %19, i32 %.scoreMax.0.18
  %cmp14 = icmp eq i32 %.scoreMax.0.19, -1
  br i1 %cmp14, label %cleanup, label %if.end16

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.034 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %src1, i32 %i.034
  %20 = load i32, ptr %arrayidx, align 4, !tbaa !3
  %arrayidx1 = getelementptr inbounds i32, ptr %src2, i32 %i.034
  %21 = load i32, ptr %arrayidx1, align 4, !tbaa !3
  %add = add nsw i32 %21, %20
  %arrayidx2 = getelementptr inbounds i32, ptr %score, i32 %i.034
  store i32 %add, ptr %arrayidx2, align 4, !tbaa !3
  %inc = add nuw nsw i32 %i.034, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.body7.preheader.loopexit, label %for.body

if.end16:                                         ; preds = %for.body7.preheader
  store i32 %.scoreMax.0.19, ptr %max, align 4, !tbaa !3
  br label %cleanup

cleanup:                                          ; preds = %for.body7.preheader, %if.end16
  %retval.0 = phi i32 [ 1, %if.end16 ], [ 0, %for.body7.preheader ]
  ret i32 %retval.0
}

attributes #0 = { norecurse nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="arm7tdmi" "target-features"="+strict-align" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"clang version 3.9.0"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
