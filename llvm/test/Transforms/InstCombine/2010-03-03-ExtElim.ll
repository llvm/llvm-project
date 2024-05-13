; RUN: opt -passes=instcombine -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-unknown-linux-gnu"

@g_92 = common global [2 x ptr] zeroinitializer, align 4 ; <ptr> [#uses=1]
@g_177 = constant ptr getelementptr (i8, ptr @g_92, i64 4), align 4 ; <ptr> [#uses=1]

define i1 @PR6486() nounwind {
; CHECK-LABEL: @PR6486(
  %tmp = load ptr, ptr @g_177                       ; <ptr> [#uses=1]
  %cmp = icmp ne ptr null, %tmp                 ; <i1> [#uses=1]
  %conv = zext i1 %cmp to i32                     ; <i32> [#uses=1]
  %cmp1 = icmp sle i32 0, %conv                   ; <i1> [#uses=1]
  ret i1 %cmp1
; CHECK: ret i1 true
}

@d = global i32 0, align 4
@a = global [1 x i32] zeroinitializer, align 4

define i1 @PR16462_1() nounwind {
; CHECK-LABEL: @PR16462_1(
  %constexpr = select i1 icmp eq (ptr @a, ptr @d), i32 0, i32 1
  %constexpr1 = trunc i32 %constexpr to i16
  %constexpr2 = sext i16 %constexpr1 to i32
  %constexpr3 = icmp sgt i32 %constexpr2, 65535
  ret i1 %constexpr3
; CHECK: ret i1 false
}

define i1 @PR16462_2() nounwind {
; CHECK-LABEL: @PR16462_2(
  %constexpr = select i1 icmp eq (ptr @a, ptr @d), i32 0, i32 1
  %constexpr1 = trunc i32 %constexpr to i16
  %constexpr2 = icmp sgt i16 %constexpr1, 42
  ret i1 %constexpr2
; CHECK: ret i1 false
}
