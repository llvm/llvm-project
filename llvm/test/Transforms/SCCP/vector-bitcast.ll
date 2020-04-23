; RUN: opt -sccp -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"

; CHECK: store volatile <2 x i64> zeroinitializer, <2 x i64>* %p
; rdar://11324230

define void @foo(<2 x i64>* %p) nounwind {
entry:
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %entry
  %vWorkExponent.i.033 = phi <4 x i32> [ %sub.i.i, %while.body.i ], [ <i32 939524096, i32 939524096, i32 939524096, i32 939524096>, %entry ]
  %sub.i.i = add <4 x i32> %vWorkExponent.i.033, <i32 -8388608, i32 -8388608, i32 -8388608, i32 -8388608>
  %0 = bitcast <4 x i32> %sub.i.i to <2 x i64>
  %and.i119.i = and <2 x i64> %0, zeroinitializer
  store volatile <2 x i64> %and.i119.i, <2 x i64>* %p
  br label %while.body.i
}

