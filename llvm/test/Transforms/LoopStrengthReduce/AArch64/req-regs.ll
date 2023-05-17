; RUN: llc -mcpu=cyclone -debug-only=loop-reduce < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; LSR used to fail here due to a bug in the ReqRegs test.
; CHECK: The chosen solution requires
; CHECK-NOT: No Satisfactory Solution

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

define void @do_integer_add(i64 %iterations, ptr nocapture readonly %cookie) {
entry:
  %0 = load i32, ptr %cookie, align 4
  %add = add nsw i32 %0, 57
  %cmp56 = icmp eq i64 %iterations, 0
  br i1 %cmp56, label %while.end, label %for.cond.preheader.preheader

for.cond.preheader.preheader:                     ; preds = %entry
  br label %for.cond.preheader

while.cond.loopexit:                              ; preds = %for.body
  %add21.lcssa = phi i32 [ %add21, %for.body ]
  %dec58 = add i64 %dec58.in, -1
  %cmp = icmp eq i64 %dec58, 0
  br i1 %cmp, label %while.end.loopexit, label %for.cond.preheader

for.cond.preheader:                               ; preds = %for.cond.preheader.preheader, %while.cond.loopexit
  %dec58.in = phi i64 [ %dec58, %while.cond.loopexit ], [ %iterations, %for.cond.preheader.preheader ]
  %a.057 = phi i32 [ %add21.lcssa, %while.cond.loopexit ], [ %add, %for.cond.preheader.preheader ]
  br label %for.body

for.body:                                         ; preds = %for.body, %for.cond.preheader
  %a.154 = phi i32 [ %a.057, %for.cond.preheader ], [ %add21, %for.body ]
  %i.053 = phi i32 [ 1, %for.cond.preheader ], [ %inc, %for.body ]
  %inc = add nsw i32 %i.053, 1
  %add2 = shl i32 %a.154, 1
  %add3 = add nsw i32 %add2, %i.053
  %add4 = shl i32 %add3, 1
  %add5 = add nsw i32 %add4, %i.053
  %add6 = shl i32 %add5, 1
  %add7 = add nsw i32 %add6, %i.053
  %add8 = shl i32 %add7, 1
  %add9 = add nsw i32 %add8, %i.053
  %add10 = shl i32 %add9, 1
  %add11 = add nsw i32 %add10, %i.053
  %add12 = shl i32 %add11, 1
  %add13 = add nsw i32 %add12, %i.053
  %add14 = shl i32 %add13, 1
  %add15 = add nsw i32 %add14, %i.053
  %add16 = shl i32 %add15, 1
  %add17 = add nsw i32 %add16, %i.053
  %add18 = shl i32 %add17, 1
  %add19 = add nsw i32 %add18, %i.053
  %add20 = shl i32 %add19, 1
  %add21 = add nsw i32 %add20, %i.053
  %exitcond = icmp eq i32 %inc, 1001
  br i1 %exitcond, label %while.cond.loopexit, label %for.body

while.end.loopexit:                               ; preds = %while.cond.loopexit
  %add21.lcssa.lcssa = phi i32 [ %add21.lcssa, %while.cond.loopexit ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %a.0.lcssa = phi i32 [ %add, %entry ], [ %add21.lcssa.lcssa, %while.end.loopexit ]
  tail call void @use_int(i32 %a.0.lcssa)
  ret void
}

declare void @use_int(i32)
