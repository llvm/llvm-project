; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer
@C = common global [100 x [100 x i32]] zeroinitializer
@D = common global [100 x [100 x [100 x i32]]] zeroinitializer

;; Test that a flow dependency in outer loop doesn't prevent interchange in
;; loops i and j.
;;
;;  for (int k = 0; k < 100; ++k) {
;;    T[k] = fn1();
;;    for (int i = 0; i < 1000; ++i)
;;      for(int j = 1; j < 1000; ++j)
;;        Arr[j][i] = Arr[j][i]+k;
;;    fn2(T[k]);
;;  }

@T = internal global [100 x double] zeroinitializer, align 4
@Arr = internal global [1000 x [1000 x i32]] zeroinitializer, align 4

define void @interchange_09(i32 %k) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  ret void

for.body:                                         ; preds = %for.cond.cleanup4, %entry
  %indvars.iv45 = phi i64 [ 0, %entry ], [ %indvars.iv.next46, %for.cond.cleanup4 ]
  %call = call double @fn1()
  %arrayidx = getelementptr inbounds [100 x double], [100 x double]* @T, i64 0, i64 %indvars.iv45
  store double %call, double* %arrayidx, align 8
  br label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %for.cond.cleanup8, %for.body
  %indvars.iv42 = phi i64 [ 0, %for.body ], [ %indvars.iv.next43, %for.cond.cleanup8 ]
  br label %for.body9

for.cond.cleanup4:                                ; preds = %for.cond.cleanup8
  %tmp = load double, double* %arrayidx, align 8
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
  %arrayidx13 = getelementptr inbounds [1000 x [1000 x i32]], [1000 x [1000 x i32]]* @Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv42
  %tmp1 = load i32, i32* %arrayidx13, align 4
  %tmp2 = trunc i64 %indvars.iv45 to i32
  %add = add nsw i32 %tmp1, %tmp2
  store i32 %add, i32* %arrayidx13, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.body9, label %for.cond.cleanup8
}

declare double @fn1() readnone
declare void @fn2(double) readnone


;; After interchange %indvars.iv (j) should increment as the middle loop.
;; After interchange %indvars.iv42 (i) should increment with the inner most loop.

; CHECK-LABEL: @interchange_09

; CHECK: for.body:
; CHECK:   %indvars.iv45 = phi i64 [ %indvars.iv.next46, %for.cond.cleanup4 ], [ 0, %for.body.preheader ]
; CHECK:   %call = call double @fn1()
; CHECK:   %arrayidx = getelementptr inbounds [100 x double], [100 x double]* @T, i64 0, i64 %indvars.iv45
; CHECK:   store double %call, double* %arrayidx, align 8
; CHECK:   br label %for.body9.preheader

; CHECK: for.cond6.preheader.preheader:
; CHECK:   br label %for.cond6.preheader

; CHECK: for.cond6.preheader:
; CHECK:   %indvars.iv42 = phi i64 [ %indvars.iv.next43, %for.cond.cleanup8 ], [ 0, %for.cond6.preheader.preheader ]
; CHECK:   br label %for.body9.split1

; CHECK: for.body9.preheader:
; CHECK:   br label %for.body9

; CHECK: for.cond.cleanup4:
; CHECK:   %tmp = load double, double* %arrayidx, align 8
; CHECK:   call void @fn2(double %tmp)
; CHECK:   %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
; CHECK:   %exitcond47 = icmp ne i64 %indvars.iv.next46, 100
; CHECK:   br i1 %exitcond47, label %for.body, label %for.cond.cleanup

; CHECK: for.cond.cleanup8:
; CHECK:   %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
; CHECK:   %exitcond44 = icmp ne i64 %indvars.iv.next43, 1000
; CHECK:   br i1 %exitcond44, label %for.cond6.preheader, label %for.body9.split

; CHECK: for.body9:
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body9.split ], [ 1, %for.body9.preheader ]
; CHECK:   br label %for.cond6.preheader.preheader

; CHECK: for.body9.split1:
; CHECK:   %arrayidx13 = getelementptr inbounds [1000 x [1000 x i32]], [1000 x [1000 x i32]]* @Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv42
; CHECK:   store i32 %add, i32* %arrayidx13, align 4
; CHECK:   br label %for.cond.cleanup8

; CHECK: for.body9.split:
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %exitcond = icmp ne i64 %indvars.iv.next, 1000
; CHECK:   br i1 %exitcond, label %for.body9, label %for.cond.cleanup4
