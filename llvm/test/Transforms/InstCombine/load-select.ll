; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"

@a = constant [2 x i32] [i32 3, i32 6]            ; <ptr> [#uses=2]

define i32 @b(i32 %y) nounwind readonly {
; CHECK-LABEL: @b(
; CHECK-NOT: load
; CHECK: ret i32
entry:
  %0 = icmp eq i32 %y, 0                          ; <i1> [#uses=1]
  %storemerge = select i1 %0, ptr getelementptr inbounds ([2 x i32], ptr @a, i32 0, i32 1), ptr @a ; <ptr> [#uses=1]
  %1 = load i32, ptr %storemerge, align 4             ; <i32> [#uses=1]
  ret i32 %1
}
