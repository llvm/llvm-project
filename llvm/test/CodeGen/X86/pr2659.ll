; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-apple-darwin9.4.0 | grep movl | count 5
; PR2659

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9.4.0"

define i32 @binomial(i32 %n, i32 %k) nounwind  {
entry:
  %cmp = icmp ugt i32 %k, %n            ; <i1> [#uses=1]
  br i1 %cmp, label %ifthen, label %forcond.preheader

forcond.preheader:              ; preds = %entry
  %cmp44 = icmp eq i32 %k, 0            ; <i1> [#uses=1]
  br i1 %cmp44, label %afterfor, label %forbody

ifthen:         ; preds = %entry
  ret i32 0

forbody:                ; preds = %forbody, %forcond.preheader
  %indvar = phi i32 [ 0, %forcond.preheader ], [ %divisor.02, %forbody ]                ; <i32> [#uses=3]
  %accumulator.01 = phi i32 [ 1, %forcond.preheader ], [ %div, %forbody ]               ; <i32> [#uses=1]
  %divisor.02 = add i32 %indvar, 1              ; <i32> [#uses=2]
  %n.addr.03 = sub i32 %n, %indvar              ; <i32> [#uses=1]
  %mul = mul i32 %n.addr.03, %accumulator.01            ; <i32> [#uses=1]
  %div = udiv i32 %mul, %divisor.02             ; <i32> [#uses=2]
  %inc = add i32 %indvar, 2             ; <i32> [#uses=1]
  %cmp4 = icmp ugt i32 %inc, %k         ; <i1> [#uses=1]
  br i1 %cmp4, label %afterfor, label %forbody

afterfor:               ; preds = %forbody, %forcond.preheader
  %accumulator.0.lcssa = phi i32 [ 1, %forcond.preheader ], [ %div, %forbody ]          ; <i32> [#uses=1]
  ret i32 %accumulator.0.lcssa
}
