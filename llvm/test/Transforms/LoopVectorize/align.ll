; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we output the abi alignment if no alignment is specified.

;CHECK-LABEL: @align
;CHECK: load <4 x i32>, <4 x i32>* {{.*}} align  4
;CHECK: load <4 x i32>, <4 x i32>* {{.*}} align  4
;CHECK: store <4 x i32> {{.*}} align  4

define void @align(i32* %a, i32* %b, i32* %c) nounwind uwtable ssp {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %3 = load i32, i32* %2
  %4 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  %5 = load i32, i32* %4
  %6 = add nsw i32 %5, %3
  %7 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %6, i32* %7
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 128
  br i1 %exitcond, label %8, label %1

; <label>:8                                       ; preds = %1
  ret void
}

