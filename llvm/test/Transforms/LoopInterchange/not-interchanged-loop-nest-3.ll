; REQUIRES: asserts
; RUN: opt < %s -basic-aa -loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@D = common global [100 x [100 x [100 x i32]]] zeroinitializer

;; Test for interchange in loop nest greater than 2.
;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      for(int k=0;k<100;k++)
;;        D[i][k][j] = D[i][k][j]+t;

; CHECK: Processing Inner Loop Id = 2 and OuterLoopId = 1
; CHECK: Loops interchanged.

; CHECK: Processing Inner Loop Id = 1 and OuterLoopId = 0
; CHECK: Interchanging loops not profitable.

define void @interchange_08(i32 %t){
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc15, %entry
  %i.028 = phi i32 [ 0, %entry ], [ %inc16, %for.inc15 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc12, %for.cond1.preheader
  %j.027 = phi i32 [ 0, %for.cond1.preheader ], [ %inc13, %for.inc12 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %k.026 = phi i32 [ 0, %for.cond4.preheader ], [ %inc, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* @D, i32 0, i32 %i.028, i32 %k.026, i32 %j.027
  %0 = load i32, i32* %arrayidx8
  %add = add nsw i32 %0, %t
  store i32 %add, i32* %arrayidx8
  %inc = add nuw nsw i32 %k.026, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.inc12, label %for.body6

for.inc12:                                        ; preds = %for.body6
  %inc13 = add nuw nsw i32 %j.027, 1
  %exitcond29 = icmp eq i32 %inc13, 100
  br i1 %exitcond29, label %for.inc15, label %for.cond4.preheader

for.inc15:                                        ; preds = %for.inc12
  %inc16 = add nuw nsw i32 %i.028, 1
  %exitcond30 = icmp eq i32 %inc16, 100
  br i1 %exitcond30, label %for.end17, label %for.cond1.preheader

for.end17:                                        ; preds = %for.inc15
  ret void
}
