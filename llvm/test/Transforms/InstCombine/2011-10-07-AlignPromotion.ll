; RUN: opt -opaque-pointers=0 -S -passes=instcombine < %s | FileCheck %s
; rdar://problem/10063307
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios5.0.0"

%0 = type { [2 x i32] }
%struct.CGPoint = type { float, float }

define void @t(%struct.CGPoint* %a) nounwind {
  %Point = alloca %struct.CGPoint, align 4
  %1 = bitcast %struct.CGPoint* %a to i64*
  %2 = bitcast %struct.CGPoint* %Point to i64*
  %3 = load i64, i64* %1, align 4
  store i64 %3, i64* %2, align 4
  call void @foo(i64* %2) nounwind
  ret void
; CHECK: %Point = alloca i64, align 4
}

declare void @foo(i64*)
