; RUN: llc -verify-machineinstrs -O0 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@c = external global i32, align 4
@d = external global [2 x i32], align 4

; Function Attrs: norecurse nounwind
define void @fn2() #0 {
; CHECK-LABEL: @fn2

  br i1 undef, label %1, label %10

; <label>:1:                                      ; preds = %0
  br i1 undef, label %3, label %2

; <label>:2:                                      ; preds = %2, %1
  br i1 undef, label %3, label %2

; <label>:3:                                      ; preds = %2, %1
  br i1 undef, label %8, label %4

; <label>:4:                                      ; preds = %4, %3
  %5 = phi i64 [ %6, %4 ], [ undef, %3 ]
  %6 = and i64 %5, and (i64 and (i64 and (i64 and (i64 and (i64 and (i64 and (i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64)), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64)), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64)), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64)), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64)), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64)), i64 sext (i32 select (i1 icmp slt (i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i16), i16 0), i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 lshr (i32 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @d, i64 0, i64 1), i32* @c) to i32), i32 6)) to i64))
  %7 = icmp slt i32 undef, 6
  br i1 %7, label %4, label %8

; <label>:8:                                      ; preds = %4, %3
  %9 = phi i64 [ undef, %3 ], [ %6, %4 ]
  br label %10

; <label>:10:                                     ; preds = %8, %0
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="ppc64le" }

