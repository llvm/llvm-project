; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@c = external global i32, align 4
@d = external global [2 x i32], align 4

; Function Attrs: norecurse nounwind
define void @fn2() #0 {
; CHECK-LABEL: @fn2
  br i1 undef, label %1, label %10

1:                                                ; preds = %0
  br i1 undef, label %3, label %2

2:                                                ; preds = %2, %1
  br i1 undef, label %3, label %2

3:                                                ; preds = %2, %1
  br i1 undef, label %8, label %4

4:                                                ; preds = %4, %3
  %5 = phi i64 [ %6, %4 ], [ undef, %3 ]
  %constexpr = getelementptr inbounds [2 x i32], ptr @d, i64 0, i64 1
  %constexpr1 = icmp eq ptr %constexpr, @c
  %constexpr2 = zext i1 %constexpr1 to i32
  %constexpr3 = getelementptr inbounds [2 x i32], ptr @d, i64 0, i64 1
  %constexpr4 = icmp eq ptr %constexpr3, @c
  %constexpr5 = zext i1 %constexpr4 to i32
  %constexpr6 = lshr i32 %constexpr5, 6
  %constexpr7 = getelementptr inbounds [2 x i32], ptr @d, i64 0, i64 1
  %constexpr8 = icmp eq ptr %constexpr7, @c
  %constexpr9 = zext i1 %constexpr8 to i16
  %constexpr10 = icmp slt i16 %constexpr9, 0
  %constexpr11 = select i1 %constexpr10, i32 %constexpr2, i32 %constexpr6
  %constexpr112 = sext i32 %constexpr11 to i64
  %constexpr213 = and i64 %constexpr112, %constexpr112
  %constexpr314 = and i64 %constexpr213, %constexpr112
  %constexpr415 = and i64 %constexpr314, %constexpr112
  %constexpr516 = and i64 %constexpr415, %constexpr112
  %constexpr617 = and i64 %constexpr516, %constexpr112
  %constexpr718 = and i64 %constexpr617, %constexpr112
  %constexpr819 = and i64 %constexpr718, %constexpr112
  %6 = and i64 %5, %constexpr819
  %7 = icmp slt i32 undef, 6
  br i1 %7, label %4, label %8

8:                                                ; preds = %4, %3
  %9 = phi i64 [ undef, %3 ], [ %6, %4 ]
  br label %10

10:                                               ; preds = %8, %0
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="ppc64le" }

