; PR41445: This test checks the case when LSR split critical edge
; and phi node has other pending fixup operands

; RUN: opt -S -loop-reduce < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; We have %indvars.iv.lcssa phi node where 4 input operands
; need to be rewritten: %tmp1, %tmp2, %tmp3, %tmp4.
; When we try to rewrite %tmp1, we first split the critical edge.
; All the other PHI inputs besides %tmp1 go to a new phi node.
; This test checks that LSR is still able to rewrite %tmp2, %tmp3, %tmp4.
define i32 @foo(i32* %A, i32 %t) {
entry:
  br label %loop.32

loop.exit:                                        ; preds = %then.8.1, %then.8, %ifmerge.42, %ifmerge.38, %ifmerge.34, %then.34
  %indvars.iv.lcssa = phi i64 [ 48, %then.8 ], [ 49, %then.8.1 ], [ %tmp4, %ifmerge.42 ], [ %tmp3, %ifmerge.38 ], [ %tmp2, %ifmerge.34 ], [ %tmp1, %then.34 ]
  %tmp = trunc i64 %indvars.iv.lcssa to i32
  br label %for.end

for.end:                                          ; preds = %then.8.1, %ifmerge.8, %loop.exit
  %i.0.lcssa = phi i32 [ %tmp, %loop.exit ], [ 50, %then.8.1 ], [ 50, %ifmerge.8 ]
  ret i32 %i.0.lcssa

; shl instruction will be dead eliminated when all it's uses will be rewritten.
; CHECK-LABEL: loop.32:
; CHECK-NOT: shl
loop.32:                                          ; preds = %ifmerge.46, %entry
  %i1.i64.0 = phi i64 [ 0, %entry ], [ %nextivloop.32, %ifmerge.46 ]
  %tmp1 = shl i64 %i1.i64.0, 2
  %tmp2 = or i64 %tmp1, 1
  %arrayIdx = getelementptr inbounds i32, i32* %A, i64 %tmp2
  %gepload = load i32, i32* %arrayIdx, align 4
  %cmp.34 = icmp sgt i32 %gepload, %t
  br i1 %cmp.34, label %then.34, label %ifmerge.34

; CHECK-LABEL: then.34:
then.34:                                          ; preds = %loop.32
  %arrayIdx17 = getelementptr inbounds i32, i32* %A, i64 %tmp1
  %gepload18 = load i32, i32* %arrayIdx17, align 4
  %cmp.35 = icmp slt i32 %gepload18, %t
  br i1 %cmp.35, label %loop.exit, label %ifmerge.34

ifmerge.34:                                       ; preds = %then.34, %loop.32
  %tmp3 = or i64 %tmp1, 2
  %arrayIdx19 = getelementptr inbounds i32, i32* %A, i64 %tmp3
  %gepload20 = load i32, i32* %arrayIdx19, align 4
  %cmp.38 = icmp sgt i32 %gepload20, %t
  %cmp.39 = icmp slt i32 %gepload, %t
  %or.cond = and i1 %cmp.38, %cmp.39
  br i1 %or.cond, label %loop.exit, label %ifmerge.38

ifmerge.38:                                       ; preds = %ifmerge.34
  %tmp4 = or i64 %tmp1, 3
  %arrayIdx23 = getelementptr inbounds i32, i32* %A, i64 %tmp4
  %gepload24 = load i32, i32* %arrayIdx23, align 4
  %cmp.42 = icmp sgt i32 %gepload24, %t
  %cmp.43 = icmp slt i32 %gepload20, %t
  %or.cond55 = and i1 %cmp.42, %cmp.43
  br i1 %or.cond55, label %loop.exit, label %ifmerge.42

ifmerge.42:                                       ; preds = %ifmerge.38
  %tmp5 = add i64 %tmp1, 4
  %arrayIdx27 = getelementptr inbounds i32, i32* %A, i64 %tmp5
  %gepload28 = load i32, i32* %arrayIdx27, align 4
  %cmp.46 = icmp sgt i32 %gepload28, %t
  %cmp.47 = icmp slt i32 %gepload24, %t
  %or.cond56 = and i1 %cmp.46, %cmp.47
  br i1 %or.cond56, label %loop.exit, label %ifmerge.46

ifmerge.46:                                       ; preds = %ifmerge.42
  %nextivloop.32 = add nuw nsw i64 %i1.i64.0, 1
  %condloop.32 = icmp ult i64 %nextivloop.32, 12
  br i1 %condloop.32, label %loop.32, label %loop.25

loop.25:                                          ; preds = %ifmerge.46
  %arrayIdx31 = getelementptr inbounds i32, i32* %A, i64 49
  %gepload32 = load i32, i32* %arrayIdx31, align 4
  %cmp.8 = icmp sgt i32 %gepload32, %t
  br i1 %cmp.8, label %then.8, label %ifmerge.8

then.8:                                           ; preds = %loop.25
  %arrayIdx33 = getelementptr inbounds i32, i32* %A, i64 48
  %gepload34 = load i32, i32* %arrayIdx33, align 4
  %cmp.15 = icmp slt i32 %gepload34, %t
  br i1 %cmp.15, label %loop.exit, label %ifmerge.8

ifmerge.8:                                        ; preds = %then.8, %loop.25
  %arrayIdx31.1 = getelementptr inbounds i32, i32* %A, i64 50
  %gepload32.1 = load i32, i32* %arrayIdx31.1, align 4
  %cmp.8.1 = icmp sgt i32 %gepload32.1, %t
  br i1 %cmp.8.1, label %then.8.1, label %for.end

then.8.1:                                         ; preds = %ifmerge.8
  %arrayIdx33.1 = getelementptr inbounds i32, i32* %A, i64 49
  %gepload34.1 = load i32, i32* %arrayIdx33.1, align 4
  %cmp.15.1 = icmp slt i32 %gepload34.1, %t
  br i1 %cmp.15.1, label %loop.exit, label %for.end
}
