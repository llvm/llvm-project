; Remove 'S' Scalar Dependencies #119345
; Scalar dependencies are not handled correctly, so they were removed to avoid
; miscompiles. The loop nest in this test case used to be interchanged, but it's
; no longer triggering. XFAIL'ing this test to indicate that this test should
; interchanged if scalar deps are handled correctly.
;
; XFAIL: *

; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer
@C = common global [100 x [100 x i32]] zeroinitializer
@D = common global [100 x [100 x [100 x i32]]] zeroinitializer
@T = internal global [100 x double] zeroinitializer, align 4
@Arr = internal global [1000 x [1000 x i32]] zeroinitializer, align 4

; Test that a flow dependency in outer loop doesn't prevent interchange in
; loops i and j.
;
;  for (int k = 0; k < 100; ++k) {
;    T[k] = fn1();
;    for (int i = 0; i < 1000; ++i)
;      for(int j = 1; j < 1000; ++j)
;        Arr[j][i] = Arr[j][i]+k;
;    fn2(T[k]);
;  }
;
; So, loops InnerLoopId = 2 and OuterLoopId = 1 should be interchanged,
; but not InnerLoopId = 1 and OuterLoopId = 0.
;
; CHECK:       --- !Passed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            Interchanged
; CHECK-NEXT:  Function:        interchange_09
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Loop interchanged with enclosing loop.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            NotTightlyNested
; CHECK-NEXT:  Function:        interchange_09
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Cannot interchange loops because they are not tightly nested.
; CHECK-NEXT:  ...
; CHECK-NEXT:  --- !Missed
; CHECK-NEXT:  Pass:            loop-interchange
; CHECK-NEXT:  Name:            InterchangeNotProfitable
; CHECK-NEXT:  Function:        interchange_09
; CHECK-NEXT:  Args:
; CHECK-NEXT:    - String:          Interchanging loops is not considered to improve cache locality nor vectorization.
; CHECK-NEXT:  ...

define void @interchange_09(i32 %k) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  ret void

for.body:                                         ; preds = %for.cond.cleanup4, %entry
  %indvars.iv45 = phi i64 [ 0, %entry ], [ %indvars.iv.next46, %for.cond.cleanup4 ]
  %call = call double @fn1()
  %arrayidx = getelementptr inbounds [100 x double], ptr @T, i64 0, i64 %indvars.iv45
  store double %call, ptr %arrayidx, align 8
  br label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %for.cond.cleanup8, %for.body
  %indvars.iv42 = phi i64 [ 0, %for.body ], [ %indvars.iv.next43, %for.cond.cleanup8 ]
  br label %for.body9

for.cond.cleanup4:                                ; preds = %for.cond.cleanup8
  %tmp = load double, ptr %arrayidx, align 8
  call void @fn2(double %tmp)
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next46, 100
  br i1 %exitcond47, label %for.body, label %for.cond.cleanup

for.cond.cleanup8:                                ; preds = %for.body9
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond44 = icmp ne i64 %indvars.iv.next43, 1000
  br i1 %exitcond44, label %for.cond6.preheader, label %for.cond.cleanup4

for.body9:                                        ; preds = %for.body9, %for.cond6.preheader
  %indvars.iv = phi i64 [ 1, %for.cond6.preheader ], [ %indvars.iv.next, %for.body9 ]
  %arrayidx13 = getelementptr inbounds [1000 x [1000 x i32]], ptr @Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv42
  %t1 = load i32, ptr %arrayidx13, align 4
  %t2 = trunc i64 %indvars.iv45 to i32
  %add = add nsw i32 %t1, %t2
  store i32 %add, ptr %arrayidx13, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.body9, label %for.cond.cleanup8
}

declare double @fn1() readnone
declare void @fn2(double) readnone
