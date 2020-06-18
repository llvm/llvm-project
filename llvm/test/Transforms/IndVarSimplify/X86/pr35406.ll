; RUN: opt -S -indvars %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

define i32 @testDiv(i8* %p, i64* %p1) {
; CHECK-LABEL: @testDiv
entry:
  br label %loop1

loop1:
  %local_0_ = phi i32 [ 8, %entry ], [ %9, %loop2.exit ]
  %local_2_ = phi i32 [ 63864, %entry ], [ %local_2_43, %loop2.exit ]
  %local_3_ = phi i32 [ 51, %entry ], [ %local_3_44, %loop2.exit ]
; CHECK-NOT:  udiv
  %0 = udiv i32 14, %local_0_
  %1 = icmp ugt i32 %local_0_, 14
  br i1 %1, label %exit, label %general_case24

; CHECK-LABEL: general_case24
general_case24:
  %2 = udiv i32 60392, %0
  br i1 false, label %loop2, label %loop2.exit

loop2:
  %local_1_56 = phi i32 [ %2, %general_case24 ], [ %3, %loop2 ]
  %local_2_57 = phi i32 [ 1, %general_case24 ], [ %7, %loop2 ]
  %3 = add i32 %local_1_56, -1
  %4 = load atomic i64, i64* %p1 unordered, align 8
  %5 = sext i32 %3 to i64
  %6 = sub i64 %4, %5
  store atomic i64 %6, i64* %p1 unordered, align 8
  %7 = add nuw nsw i32 %local_2_57, 1
  %8 = icmp ugt i32 %local_2_57, 7
  br i1 %8, label %loop2.exit, label %loop2

loop2.exit:
  %local_2_43 = phi i32 [ %local_2_, %general_case24 ], [ 9, %loop2 ]
  %local_3_44 = phi i32 [ %local_3_, %general_case24 ], [ %local_1_56, %loop2 ]
  %9 = add nuw nsw i32 %local_0_, 1
  %10 = icmp ugt i32 %local_0_, 129
  br i1 %10, label %exit, label %loop1

exit:
  ret i32 0
}

define i32 @testRem(i8* %p, i64* %p1) {
; CHECK-LABEL: @testRem
entry:
  br label %loop1

loop1:
  %local_0_ = phi i32 [ 8, %entry ], [ %9, %loop2.exit ]
  %local_2_ = phi i32 [ 63864, %entry ], [ %local_2_43, %loop2.exit ]
  %local_3_ = phi i32 [ 51, %entry ], [ %local_3_44, %loop2.exit ]
; CHECK:  udiv
; CHECK-NOT:  udiv
  %0 = udiv i32 14, %local_0_
  %1 = icmp ugt i32 %local_0_, 14
  br i1 %1, label %exit, label %general_case24

; CHECK-LABEL: general_case24
general_case24:
  %2 = urem i32 60392, %0
  br i1 false, label %loop2, label %loop2.exit

loop2:
  %local_1_56 = phi i32 [ %2, %general_case24 ], [ %3, %loop2 ]
  %local_2_57 = phi i32 [ 1, %general_case24 ], [ %7, %loop2 ]
  %3 = add i32 %local_1_56, -1
  %4 = load atomic i64, i64* %p1 unordered, align 8
  %5 = sext i32 %3 to i64
  %6 = sub i64 %4, %5
  store atomic i64 %6, i64* %p1 unordered, align 8
  %7 = add nuw nsw i32 %local_2_57, 1
  %8 = icmp ugt i32 %local_2_57, 7
  br i1 %8, label %loop2.exit, label %loop2

loop2.exit:
  %local_2_43 = phi i32 [ %local_2_, %general_case24 ], [ 9, %loop2 ]
  %local_3_44 = phi i32 [ %local_3_, %general_case24 ], [ %local_1_56, %loop2 ]
  %9 = add nuw nsw i32 %local_0_, 1
  %10 = icmp ugt i32 %local_0_, 129
  br i1 %10, label %exit, label %loop1

exit:
  ret i32 0
}
