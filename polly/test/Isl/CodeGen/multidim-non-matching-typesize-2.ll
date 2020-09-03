; RUN: opt %loadPolly -disable-basic-aa -polly-codegen \
; RUN:     -S < %s | FileCheck %s
; CHECK: polly
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"

define void @hoge(i16* %arg, i32 %arg1, i16* %arg2) {
bb:
  %tmp = alloca i32
  %tmp3 = load i32, i32* undef
  br label %bb7

bb7:
  %tmp8 = phi i32 [ %tmp3, %bb ], [ %tmp13, %bb7 ]
  %tmp9 = getelementptr inbounds i16, i16* %arg2, i32 0
  %tmp10 = load i16, i16* %tmp9, align 2
  %tmp11 = mul nsw i32 %tmp8, %arg1
  %tmp12 = getelementptr inbounds i16, i16* %arg, i32 %tmp11
  store i16 undef, i16* %tmp12, align 2
  %tmp13 = add nsw i32 %tmp8, 1
  store i32 undef, i32* %tmp
  br i1 false, label %bb7, label %bb5

bb5:
  ret void

}
