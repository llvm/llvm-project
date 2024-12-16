; RUN: opt < %s -passes=debugify,loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -prefer-inloop-reductions -S | FileCheck %s -check-prefix DEBUGLOC
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Testing the debug locations of the generated vector intstruction are same as
; their scalar instruction.

; DEBUGLOC-LABEL: define i32 @reduction_sum(
define i32 @reduction_sum(ptr noalias nocapture %A, ptr noalias nocapture %B) {
; DEBUGLOC: vector.body:
; DEBUGLOC:   %[[VecLoad:.*]] = load <4 x i32>, ptr %2, align 4, !dbg ![[LoadLoc0:[0-9]+]]
; DEBUGLOC:   %[[VecRed:.*]] = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %wide.load), !dbg ![[LoadLoc0]]
; DEBUGLOC: .lr.ph:
; DEBUGLOC:   %l3 = load i32, ptr %l2, align 4, !dbg ![[LoadLoc0]]
; DEBUGLOC:   %l7 = add i32 %sum.02, %l3, !dbg ![[RedLoc0:[0-9]+]]
entry:
  br label %.lr.ph

.lr.ph:                                           ; preds = %entry, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %entry ]
  %sum.02 = phi i32 [ %l7, %.lr.ph ], [ 0, %entry ]
  %l2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %l3 = load i32, ptr %l2, align 4
  %l6 = trunc i64 %indvars.iv to i32
  %l7 = add i32 %sum.02, %l3
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph
  %sum.0.lcssa = phi i32 [ %l7, %.lr.ph ]
  ret i32 %sum.0.lcssa
}
