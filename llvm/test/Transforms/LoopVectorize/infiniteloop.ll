; RUN: opt -S -passes='loop(indvars),loop-vectorize' -force-vector-width=2  < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

@a = common global i64 0, align 8
@x = common global i32 0, align 4

; We used to assert on this loop because we could not find an induction
; variable but assumed there must be one. Scalar evolution returned a exit
; count for the loop below and from there on we assumed that there must be an
; induction variable. This is not a valid assumption:
;   // getExitCount - Get the expression for the number of loop iterations for
;   // which this loop is *guaranteed not to exit* via ExitingBlock. Otherwise
;   // return SCEVCouldNotCompute.
; For an infinite loop SE can return any number.

; CHECK-LABEL: @fn1(
define void @fn1()  {
entry:
  store i64 0, ptr @a, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %inc1 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  store volatile i32 0, ptr @x, align 4
  %inc = add nsw i64 %inc1, 1
  %cmp = icmp sgt i64 %inc1, -2
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %inc.lcssa = phi i64 [ %inc, %for.body ]
  store i64 %inc.lcssa, ptr @a, align 8
  ret void
}
