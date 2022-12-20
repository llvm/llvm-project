; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -debug-only=loop-vectorize\
; RUN:     -disable-output -S %s 2>&1 | FileCheck %s

; Make sure that we report about not vectorizing functions with 'noimplicitfloat' attributes

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16

; CHECK: LV: Not vectorizing: Can't vectorize when the NoImplicitFloat attribute is used
define void @example_nofloat() noimplicitfloat { ;           <--------- "noimplicitfloat" attribute here!
  br label %1

; <label>:1                                       ; preds = %1, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %1 ]
  %2 = getelementptr inbounds [2048 x i32], ptr @a, i64 0, i64 %indvars.iv
  %3 = trunc i64 %indvars.iv to i32
  store i32 %3, ptr %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %4, label %1

; <label>:4                                       ; preds = %1
  ret void
}
