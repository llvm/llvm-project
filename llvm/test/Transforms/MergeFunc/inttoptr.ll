; RUN: opt -passes=mergefunc -S < %s | FileCheck %s
; PR15185
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

%.qux.2496 = type { i32, %.qux.2497 }
%.qux.2497 = type { i8, i32 }
%.qux.2585 = type { i32, i32, ptr }

@g2 = external unnamed_addr constant [9 x i8], align 1
@g3 = internal unnamed_addr constant [1 x ptr] [ptr @func35]

define internal i32 @func1(ptr %ptr, ptr nocapture %method) align 2 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb1
  ret i32 undef
}

define internal i32 @func10(ptr nocapture %this) align 2 {
bb:
  %tmp = getelementptr inbounds %.qux.2496, ptr %this, i32 0, i32 1, i32 1
  %tmp1 = load i32, ptr %tmp, align 4
  ret i32 %tmp1
}

define internal ptr @func29(ptr nocapture %this) align 2 {
bb:
  ret ptr @g2
}

define internal ptr @func33(ptr nocapture %this) align 2 {
bb:
  ret ptr undef
}

define internal ptr @func34(ptr nocapture %this) align 2 {
bb:
  ret ptr undef
}

define internal ptr @func35(ptr nocapture %this) align 2 {
bb:
; CHECK-LABEL: @func35(
; CHECK: %[[V3:.+]] = tail call i32 @func10(ptr captures(none) %{{.*}})
; CHECK: %{{.*}} = inttoptr i32 %[[V3]] to ptr
  %tmp = getelementptr inbounds %.qux.2585, ptr %this, i32 0, i32 2
  %tmp1 = load ptr, ptr %tmp, align 4
  ret ptr %tmp1
}
