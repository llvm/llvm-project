; RUN: opt < %s -bb-vectorize -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@d = external global [1 x [10 x [1 x i16]]], align 16

;CHECK-LABEL: @test
;CHECK: %0 = select i1 %bool, <4 x i16> <i16 -2, i16 -2, i16 -2, i16 -2>, <4 x i16> <i16 -3, i16 -3, i16 -3, i16 -3>
;CHECK: %1 = select i1 %bool, <4 x i16> <i16 -2, i16 -2, i16 -2, i16 -2>, <4 x i16> <i16 -3, i16 -3, i16 -3, i16 -3>
;CHECK: %2 = shufflevector <4 x i16> %0, <4 x i16> %1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;CHECK: %3 = shufflevector <4 x i1> %boolvec, <4 x i1> %boolvec, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
;CHECK: %4 = select <8 x i1> %3, <8 x i16> <i16 -3, i16 -3, i16 -3, i16 -3, i16 -3, i16 -3, i16 -3, i16 -3>, <8 x i16> %2
define void @test() {
entry:
  %bool = icmp ne i32 undef, 0
  %boolvec = icmp ne <4 x i32> undef, zeroinitializer
  br label %body

body:
  %0 = select i1 %bool, <4 x i16> <i16 -2, i16 -2, i16 -2, i16 -2>, <4 x i16> <i16 -3, i16 -3, i16 -3, i16 -3>
  %1 = select i1 %bool, <4 x i16> <i16 -2, i16 -2, i16 -2, i16 -2>, <4 x i16> <i16 -3, i16 -3, i16 -3, i16 -3>
  %2 = select <4 x i1> %boolvec, <4 x i16> <i16 -3, i16 -3, i16 -3, i16 -3>, <4 x i16> %0
  %3 = select <4 x i1> %boolvec, <4 x i16> <i16 -3, i16 -3, i16 -3, i16 -3>, <4 x i16> %1
  %4 = add nsw <4 x i16> %2, zeroinitializer
  %5 = add nsw <4 x i16> %3, zeroinitializer
  %6 = getelementptr inbounds [1 x [10 x [1 x i16]]], [1 x [10 x [1 x i16]]]* @d, i64 0, i64 0, i64 undef, i64 0
  %7 = bitcast i16* %6 to <4 x i16>*
  store <4 x i16> %4, <4 x i16>* %7, align 2
  %8 = getelementptr [1 x [10 x [1 x i16]]], [1 x [10 x [1 x i16]]]* @d, i64 0, i64 0, i64 undef, i64 4
  %9 = bitcast i16* %8 to <4 x i16>*
  store <4 x i16> %5, <4 x i16>* %9, align 2
  ret void
}
