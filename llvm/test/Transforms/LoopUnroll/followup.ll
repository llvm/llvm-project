; Check that followup attributes are applied after LoopUnroll.
;
; We choose -unroll-count=3 because it produces partial unrolling of remainder
; loops.  Complete unrolling would leave no remainder loop to which to copy
; followup attributes.

; DEFINE: %{unroll} = opt < %s -S -passes=loop-unroll -unroll-count=3
; DEFINE: %{epilog} = %{unroll} -unroll-runtime -unroll-runtime-epilog=true
; DEFINE: %{prolog} = %{unroll} -unroll-runtime -unroll-runtime-epilog=false
; DEFINE: %{fc} = FileCheck %s -check-prefixes

; RUN: %{unroll} | %{fc} COMMON,COUNT
; RUN: %{epilog} | %{fc} COMMON,EPILOG,EPILOG-NO-UNROLL
; RUN: %{prolog} | %{fc} COMMON,PROLOG,PROLOG-NO-UNROLL
; RUN: %{epilog} -unroll-remainder | %{fc} COMMON,EPILOG,EPILOG-UNROLL
; RUN: %{prolog} -unroll-remainder | %{fc} COMMON,PROLOG,PROLOG-UNROLL

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i32 @test(ptr nocapture %a, i32 %n) nounwind uwtable readonly {
entry:
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !4

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}

!1 = !{!"llvm.loop.unroll.followup_all", !{!"FollowupAll"}}
!2 = !{!"llvm.loop.unroll.followup_unrolled", !{!"FollowupUnrolled"}}
!3 = !{!"llvm.loop.unroll.followup_remainder", !{!"FollowupRemainder"}}
!4 = distinct !{!4, !1, !2, !3}


; COMMON-LABEL: @test(


; COUNT: br i1 %exitcond.2, label %for.end.loopexit, label %for.body, !llvm.loop ![[#LOOP:]]

; COUNT: ![[#LOOP]] = distinct !{![[#LOOP]], ![[#FOLLOWUP_ALL:]], ![[#FOLLOWUP_UNROLLED:]]}
; COUNT: ![[#FOLLOWUP_ALL]] = !{!"FollowupAll"}
; COUNT: ![[#FOLLOWUP_UNROLLED]] = !{!"FollowupUnrolled"}


; EPILOG: br i1 %niter.ncmp.2, label %for.end.loopexit.unr-lcssa, label %for.body, !llvm.loop ![[LOOP_0:[0-9]+]]
; EPILOG-NO-UNROLL: br i1 %epil.iter.cmp, label %for.body.epil, label %for.end.loopexit.epilog-lcssa, !llvm.loop ![[LOOP_2:[0-9]+]]
; EPILOG-UNROLL: br i1 %epil.iter.cmp, label %for.body.epil.1, label %for.end.loopexit.epilog-lcssa
; EPILOG-UNROLL: br i1 %epil.iter.cmp.1, label %for.body.epil, label %for.end.loopexit.epilog-lcssa, !llvm.loop ![[LOOP_2:[0-9]+]]

; EPILOG: ![[LOOP_0]] = distinct !{![[LOOP_0]], ![[FOLLOWUP_ALL:[0-9]+]], ![[FOLLOWUP_UNROLLED:[0-9]+]]}
; EPILOG: ![[FOLLOWUP_ALL]] = !{!"FollowupAll"}
; EPILOG: ![[FOLLOWUP_UNROLLED]] = !{!"FollowupUnrolled"}
; EPILOG: ![[LOOP_2]] = distinct !{![[LOOP_2]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_REMAINDER:[0-9]+]]}
; EPILOG: ![[FOLLOWUP_REMAINDER]] = !{!"FollowupRemainder"}


; PROLOG-UNROLL:  br i1 %prol.iter.cmp, label %for.body.prol.1, label %for.body.prol.loopexit.unr-lcssa
; PROLOG-UNROLL:  br i1 %prol.iter.cmp.1, label %for.body.prol, label %for.body.prol.loopexit.unr-lcssa, !llvm.loop ![[LOOP_0:[0-9]+]]
; PROLOG-NO-UNROLL:  br i1 %prol.iter.cmp, label %for.body.prol, label %for.body.prol.loopexit.unr-lcssa, !llvm.loop ![[LOOP_0:[0-9]+]]
; PROLOG:  br i1 %exitcond.2, label %for.end.loopexit.unr-lcssa, label %for.body, !llvm.loop ![[LOOP_2:[0-9]+]]

; PROLOG: ![[LOOP_0]] = distinct !{![[LOOP_0]], ![[FOLLOWUP_ALL:[0-9]+]], ![[FOLLOWUP_REMAINDER:[0-9]+]]}
; PROLOG: ![[FOLLOWUP_ALL]] = !{!"FollowupAll"}
; PROLOG: ![[FOLLOWUP_REMAINDER]] = !{!"FollowupRemainder"}
; PROLOG: ![[LOOP_2]] = distinct !{![[LOOP_2]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_UNROLLED:[0-9]+]]}
; PROLOG: ![[FOLLOWUP_UNROLLED]] = !{!"FollowupUnrolled"}
