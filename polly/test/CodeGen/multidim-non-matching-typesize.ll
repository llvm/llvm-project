; RUN: opt %loadNPMPolly -disable-basic-aa -passes=polly-codegen \
; RUN:     -S < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

; CHECK: polly

define void @hoge(ptr %arg, i32 %arg1) {
bb:
  %tmp = alloca i16
  br label %bb2

bb2:
  %tmp3 = phi i32 [ %tmp7, %bb2 ], [ 0, %bb ]
  %tmp4 = mul nsw i32 %tmp3, %arg1
  %tmp5 = getelementptr inbounds i16, ptr %arg, i32 %tmp4
  %tmp6 = load i16, ptr %tmp5, align 2
  store i16 %tmp6, ptr %tmp
  %tmp7 = add nsw i32 %tmp3, 1
  br i1 false, label %bb2, label %bb8

bb8:
  ret void
}
