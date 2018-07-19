; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@D = common global [100 x [100 x [100 x i32]]] zeroinitializer

;; Test for interchange in loop nest greater than 2.
;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      for(int k=0;k<100;k++)
;;        D[i][k][j] = D[i][k][j]+t;

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
; CHECK-LABEL: @interchange_08
; CHECK:   entry:
; CHECK:     br label %for.cond1.preheader.preheader
; CHECK:   for.cond1.preheader.preheader:                    ; preds = %entry
; CHECK:     br label %for.cond1.preheader
; CHECK:   for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc15
; CHECK:     %i.028 = phi i32 [ %inc16, %for.inc15 ], [ 0, %for.cond1.preheader.preheader ]
; CHECK:     br label %for.body6.preheader
; CHECK:   for.cond4.preheader.preheader:                    ; preds = %for.body6
; CHECK:     br label %for.cond4.preheader
; CHECK:   for.cond4.preheader:                              ; preds = %for.cond4.preheader.preheader, %for.inc12
; CHECK:     %j.027 = phi i32 [ %inc13, %for.inc12 ], [ 0, %for.cond4.preheader.preheader ]
; CHECK:     br label %for.body6.split1
; CHECK:   for.body6.preheader:                              ; preds = %for.cond1.preheader
; CHECK:     br label %for.body6
; CHECK:   for.body6:                                        ; preds = %for.body6.preheader, %for.body6.split
; CHECK:     %k.026 = phi i32 [ %inc, %for.body6.split ], [ 0, %for.body6.preheader ]
; CHECK:     br label %for.cond4.preheader.preheader
; CHECK:   for.body6.split1:                                 ; preds = %for.cond4.preheader
; CHECK:     %arrayidx8 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* @D, i32 0, i32 %i.028, i32 %k.026, i32 %j.027
; CHECK:     %0 = load i32, i32* %arrayidx8
; CHECK:     %add = add nsw i32 %0, %t
; CHECK:     store i32 %add, i32* %arrayidx8
; CHECK:     br label %for.inc12
; CHECK:   for.body6.split:                                  ; preds = %for.inc12
; CHECK:     %inc = add nuw nsw i32 %k.026, 1
; CHECK:     %exitcond = icmp eq i32 %inc, 100
; CHECK:     br i1 %exitcond, label %for.inc15, label %for.body6
; CHECK:   for.inc12:                                        ; preds = %for.body6.split1
; CHECK:     %inc13 = add nuw nsw i32 %j.027, 1
; CHECK:     %exitcond29 = icmp eq i32 %inc13, 100
; CHECK:     br i1 %exitcond29, label %for.body6.split, label %for.cond4.preheader
; CHECK:   for.inc15:                                        ; preds = %for.body6.split
; CHECK:     %inc16 = add nuw nsw i32 %i.028, 1
; CHECK:     %exitcond30 = icmp eq i32 %inc16, 100
; CHECK:     br i1 %exitcond30, label %for.end17, label %for.cond1.preheader
; CHECK:   for.end17:                                        ; preds = %for.inc15
; CHECK:     ret void
