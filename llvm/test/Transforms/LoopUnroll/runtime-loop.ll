; RUN: opt < %s -S -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=true  | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -loop-unroll -unroll-runtime=true -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG

; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(unroll)' -unroll-runtime=true -unroll-runtime-epilog=true  | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -passes='require<opt-remark-emit>,loop(unroll)' -unroll-runtime=true -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Tests for unrolling loops with run-time trip counts

; EPILOG: %xtraiter = and i32 %n
; EPILOG:  %lcmp.mod = icmp ne i32 %xtraiter, 0
; EPILOG:  br i1 %lcmp.mod, label %for.body.epil.preheader, label %for.end.loopexit

; PROLOG: %xtraiter = and i32 %n
; PROLOG:  %lcmp.mod = icmp ne i32 %xtraiter, 0
; PROLOG:  br i1 %lcmp.mod, label %for.body.prol.preheader, label %for.body.prol.loopexit

; EPILOG: for.body.epil:
; EPILOG: %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %for.body.epil ],  [ %indvars.iv.unr, %for.body.epil.preheader ]
; EPILOG:  %epil.iter.sub = sub i32 %epil.iter, 1
; EPILOG:  %epil.iter.cmp = icmp ne i32 %epil.iter.sub, 0
; EPILOG:  br i1 %epil.iter.cmp, label %for.body.epil, label %for.end.loopexit.epilog-lcssa, !llvm.loop !0

; PROLOG: for.body.prol:
; PROLOG: %indvars.iv.prol = phi i64 [ %indvars.iv.next.prol, %for.body.prol ], [ 0, %for.body.prol.preheader ]
; PROLOG:  %prol.iter.sub = sub i32 %prol.iter, 1
; PROLOG:  %prol.iter.cmp = icmp ne i32 %prol.iter.sub, 0
; PROLOG:  br i1 %prol.iter.cmp, label %for.body.prol, label %for.body.prol.loopexit.unr-lcssa, !llvm.loop !0


define i32 @test(i32* nocapture %a, i32 %n) nounwind uwtable readonly {
entry:
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}


; Still try to completely unroll loops with compile-time trip counts
; even if the -unroll-runtime is specified

; EPILOG: for.body:
; EPILOG-NOT: for.body.epil:

; PROLOG: for.body:
; PROLOG-NOT: for.body.prol:

define i32 @test1(i32* nocapture %a) nounwind uwtable readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %sum.01 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.01
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 5
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 %add
}

; This is test 2007-05-09-UnknownTripCount.ll which can be unrolled now
; if the -unroll-runtime option is turned on

; EPILOG: bb72.2:
; PROLOG: bb72.2:

define void @foo(i32 %trips) {
entry:
        br label %cond_true.outer

cond_true.outer:
        %indvar1.ph = phi i32 [ 0, %entry ], [ %indvar.next2, %bb72 ]
        br label %bb72

bb72:
        %indvar.next2 = add i32 %indvar1.ph, 1
        %exitcond3 = icmp eq i32 %indvar.next2, %trips
        br i1 %exitcond3, label %cond_true138, label %cond_true.outer

cond_true138:
        ret void
}


; Test run-time unrolling for a loop that counts down by -2.

; EPILOG: for.body.epil:
; EPILOG: br i1 %epil.iter.cmp, label %for.body.epil, label %for.cond.for.end_crit_edge.epilog-lcssa

; PROLOG: for.body.prol:
; PROLOG: br i1 %prol.iter.cmp, label %for.body.prol, label %for.body.prol.loopexit

define zeroext i16 @down(i16* nocapture %p, i32 %len) nounwind uwtable readonly {
entry:
  %cmp2 = icmp eq i32 %len, 0
  br i1 %cmp2, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %p.addr.05 = phi i16* [ %incdec.ptr, %for.body ], [ %p, %entry ]
  %len.addr.04 = phi i32 [ %sub, %for.body ], [ %len, %entry ]
  %res.03 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i16, i16* %p.addr.05, i64 1
  %0 = load i16, i16* %p.addr.05, align 2
  %conv = zext i16 %0 to i32
  %add = add i32 %conv, %res.03
  %sub = add nsw i32 %len.addr.04, -2
  %cmp = icmp eq i32 %sub, 0
  br i1 %cmp, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %phitmp = trunc i32 %add to i16
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %res.0.lcssa = phi i16 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i16 %res.0.lcssa
}

; Test run-time unrolling disable metadata.
; EPILOG: for.body:
; EPILOG-NOT: for.body.epil:

; PROLOG: for.body:
; PROLOG-NOT: for.body.prol:

define zeroext i16 @test2(i16* nocapture %p, i32 %len) nounwind uwtable readonly {
entry:
  %cmp2 = icmp eq i32 %len, 0
  br i1 %cmp2, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %p.addr.05 = phi i16* [ %incdec.ptr, %for.body ], [ %p, %entry ]
  %len.addr.04 = phi i32 [ %sub, %for.body ], [ %len, %entry ]
  %res.03 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i16, i16* %p.addr.05, i64 1
  %0 = load i16, i16* %p.addr.05, align 2
  %conv = zext i16 %0 to i32
  %add = add i32 %conv, %res.03
  %sub = add nsw i32 %len.addr.04, -2
  %cmp = icmp eq i32 %sub, 0
  br i1 %cmp, label %for.cond.for.end_crit_edge, label %for.body, !llvm.loop !0

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %phitmp = trunc i32 %add to i16
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %res.0.lcssa = phi i16 [ %phitmp, %for.cond.for.end_crit_edge ], [ 0, %entry ]
  ret i16 %res.0.lcssa
}

; dont unroll loop with multiple exit/exiting blocks, unless
; -runtime-unroll-multi-exit=true
; single exit, multiple exiting blocks.
define void @unique_exit(i32 %arg) {
; PROLOG: unique_exit(
; PROLOG-NOT: .unr

; EPILOG: unique_exit(
; EPILOG-NOT: .unr
entry:
  %tmp = icmp sgt i32 undef, %arg
  br i1 %tmp, label %preheader, label %returnblock

preheader:                                 ; preds = %entry
  br label %header

LoopExit:                                ; preds = %header, %latch
  %tmp2.ph = phi i32 [ %tmp4, %header ], [ -1, %latch ]
  br label %returnblock

returnblock:                                         ; preds = %LoopExit, %entry
  %tmp2 = phi i32 [ -1, %entry ], [ %tmp2.ph, %LoopExit ]
  ret void

header:                                           ; preds = %preheader, %latch
  %tmp4 = phi i32 [ %inc, %latch ], [ %arg, %preheader ]
  %inc = add nsw i32 %tmp4, 1
  br i1 true, label %LoopExit, label %latch

latch:                                            ; preds = %header
  %cmp = icmp slt i32 %inc, undef
  br i1 %cmp, label %header, label %LoopExit
}

; multiple exit blocks. don't unroll
define void @multi_exit(i64 %trip, i1 %cond) {
; PROLOG: multi_exit(
; PROLOG-NOT: .unr

; EPILOG: multi_exit(
; EPILOG-NOT: .unr
entry:
  br label %loop_header

loop_header:
  %iv = phi i64 [ 0, %entry ], [ %iv_next, %loop_latch ]
  br i1 %cond, label %loop_latch, label %loop_exiting_bb1

loop_exiting_bb1:
  br i1 false, label %loop_exiting_bb2, label %exit1

loop_exiting_bb2:
  br i1 false, label %loop_latch, label %exit3

exit3:
  ret void

loop_latch:
  %iv_next = add i64 %iv, 1
  %cmp = icmp ne i64 %iv_next, %trip
  br i1 %cmp, label %loop_header, label %exit2.loopexit

exit1:
 ret void

exit2.loopexit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.runtime.disable"}

; EPILOG: !0 = distinct !{!0, !1}
; EPILOG: !1 = !{!"llvm.loop.unroll.disable"}

; PROLOG: !0 = distinct !{!0, !1}
; PROLOG: !1 = !{!"llvm.loop.unroll.disable"}
