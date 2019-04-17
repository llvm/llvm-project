; RUN: opt < %s -indvars -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

@data = common global [240 x i8] zeroinitializer, align 16

define void @foo(i8* %a) nounwind uwtable ssp {
; CHECK: %exitcond
; CHECK-NOT: ([240 x i8]* @data, i64 0, i64 -16)
  br label %1

; <label>:1                                       ; preds = %0, %1
  %i.0 = phi i8 [ 0, %0 ], [ %5, %1 ]
  %p.0 = phi i8* [ getelementptr inbounds ([240 x i8], [240 x i8]* @data, i64 0, i64 0), %0 ], [ %4, %1 ]
  %.0 = phi i8* [ %a, %0 ], [ %2, %1 ]
  %2 = getelementptr inbounds i8, i8* %.0, i64 1
  %3 = load i8, i8* %.0, align 1
  %4 = getelementptr inbounds i8, i8* %p.0, i64 1
  store i8 %3, i8* %p.0, align 1
  %5 = add i8 %i.0, 1
  %6 = icmp ult i8 %5, -16
  br i1 %6, label %1, label %7

; <label>:7                                       ; preds = %1
  ret void
}
