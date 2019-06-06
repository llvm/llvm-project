; RUN: opt -hexagon-loop-idiom -S < %s | FileCheck %s

; Make sure that we generate correct runtime checks.

; CHECK: b7.old:
; CHECK:   [[LEN:%[0-9]+]] = shl nuw i32 %len, 3
; CHECK:   [[SRC:%[0-9]+]] = ptrtoint i8* %src to i32
; CHECK:   [[DST:%[0-9]+]] = ptrtoint i8* %dst to i32
; CHECK:   [[ULT:%[0-9]+]] = icmp ult i32 [[DST]], [[SRC]]
; CHECK:   [[SUB:%[0-9]+]] = sub i32 [[DST]], [[SRC]]
; CHECK:   [[SLT:%[0-9]+]] = icmp sle i32 [[LEN]], [[SUB]]
; CHECK:   [[CND:%[0-9]+]] = or i1 [[ULT]], [[SLT]]
; CHECK:   br i1 [[CND]], label %b8.rtli, label %b8.rtli.ph

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred(i8* %dst, i8* %src, i32 %len) #0 {
b3:
  %v4 = bitcast i8* %src to i64*
  %v5 = bitcast i8* %dst to i64*
  %v6 = icmp slt i32 0, %len
  br i1 %v6, label %b7, label %b16

b7:                                               ; preds = %b3
  br label %b8

b8:                                               ; preds = %b13, %b7
  %v9 = phi i32 [ 0, %b7 ], [ %v14, %b13 ]
  %v10 = getelementptr inbounds i64, i64* %v4, i32 %v9
  %v11 = load i64, i64* %v10, align 8
  %v12 = getelementptr inbounds i64, i64* %v5, i32 %v9
  store i64 %v11, i64* %v12, align 8
  br label %b13

b13:                                              ; preds = %b8
  %v14 = add nsw i32 %v9, 1
  %v15 = icmp slt i32 %v14, %len
  br i1 %v15, label %b8, label %b16

b16:                                              ; preds = %b13, %b3
  ret void
}

attributes #0 = { noinline nounwind "target-cpu"="hexagonv60" }
