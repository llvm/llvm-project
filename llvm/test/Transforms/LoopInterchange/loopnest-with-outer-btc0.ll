; RUN: opt < %s -passes=loop-interchange -verify-dom-info -verify-loop-info \
; RUN:       -pass-remarks-output=%t -pass-remarks='loop-interchange' -S
; RUN: cat %t |  FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@D = common global [100 x [100 x [100 x i32]]] zeroinitializer

; The outer loop's backedge isn't taken. Check the loop with BTC=0 is considered
; unprofitable, but that we still interchange the two inner loops.
;
;  for(int i=0;i<1;i++)
;    for(int j=0;j<100;j++)
;      for(int k=0;k<100;k++)
;        D[i][k][j] = D[i][k][j]+t;
;

; CHECK:        --- !Analysis
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            Dependence
; CHECK-NEXT:   Function:        interchange_i_and_j
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          Computed dependence info, invoking the transform.
; CHECK-NEXT:   ...
; CHECK-NEXT:   --- !Passed
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            Interchanged
; CHECK-NEXT:   Function:        interchange_i_and_j
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT:   ...
; CHECK-NEXT:   --- !Missed
; CHECK-NEXT:   Pass:            loop-interchange
; CHECK-NEXT:   Name:            InterchangeNotProfitable
; CHECK-NEXT:   Function:        interchange_i_and_j
; CHECK-NEXT:   Args:
; CHECK-NEXT:     - String:          Insufficient information to calculate the cost of loop for interchange.
; CHECK-NEXT:   ...

define void @interchange_i_and_j(i32 %t){
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %inc16, %for.inc15 ]
  br label %inner1.header

inner1.header:
  %j = phi i64 [ 0, %outer.header ], [ %inc13, %for.inc12 ]
  br label %inner2.body

inner2.body:
  %k = phi i64 [ 0, %inner1.header ], [ %inc, %inner2.body ]
  %arrayidx8 = getelementptr inbounds [100 x [100 x i32]], ptr @D, i64 %i, i64 %k, i64 %j
  %0 = load i32, ptr %arrayidx8
  %add = add nsw i32 %0, %t
  store i32 %add, ptr %arrayidx8
  %inc = add nuw nsw i64 %k, 1
  %exitcond = icmp eq i64 %inc, 100
  br i1 %exitcond, label %for.inc12, label %inner2.body

for.inc12:
  %inc13 = add nuw nsw i64 %j, 1
  %exitcond29 = icmp eq i64 %inc13, 100
  br i1 %exitcond29, label %for.inc15, label %inner1.header

for.inc15:
  %inc16 = add nuw nsw i64 %i, 1
  %exitcond30 = icmp eq i64 %inc16, 1
  br i1 %exitcond30, label %for.end17, label %outer.header

for.end17:
  ret void
}
