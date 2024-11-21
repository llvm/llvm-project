; RUN: opt %loadNPMPolly -pass-remarks-analysis="polly-scops" '-passes=print<polly-function-scops>' \
; RUN:     -polly-invariant-load-hoisting=true \
; RUN:     -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Low complexity assumption:
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@board = external global [421 x i8], align 16

; Function Attrs: nounwind uwtable
define fastcc void @ping_recurse(ptr nocapture %mx, ptr nocapture %mr, i32 %color) unnamed_addr {
entry:
  br label %land.lhs.true38.1

if.end58:                                         ; preds = %land.lhs.true38.2, %if.end54.1
  ret void

land.lhs.true38.1:                                ; preds = %entry
  %0 = load i32, ptr %mr, align 4
  %cmp41.1 = icmp eq i32 %0, 0
  br i1 %cmp41.1, label %land.lhs.true43.1, label %if.end54.1

land.lhs.true43.1:                                ; preds = %land.lhs.true38.1
  %1 = load i32, ptr %mx, align 4
  %cmp46.1 = icmp eq i32 %1, 1
  %cmp51.1 = icmp eq i32 0, %color
  %or.cond.1 = or i1 %cmp51.1, %cmp46.1
  br i1 %or.cond.1, label %if.then53.1, label %if.end54.1

if.then53.1:                                      ; preds = %land.lhs.true43.1
  tail call fastcc void @ping_recurse(ptr nonnull %mx, ptr nonnull %mr, i32 %color)
  br label %if.end54.1

if.end54.1:                                       ; preds = %if.then53.1, %land.lhs.true43.1, %land.lhs.true38.1
  %2 = load i8, ptr @board, align 1
  %cmp36.2 = icmp eq i8 %2, 3
  br i1 %cmp36.2, label %if.end58, label %land.lhs.true38.2

land.lhs.true38.2:                                ; preds = %if.end54.1
  %3 = load i32, ptr %mr, align 4
  br label %if.end58
}
