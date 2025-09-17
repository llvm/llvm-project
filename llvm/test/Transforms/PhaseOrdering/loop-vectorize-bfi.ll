; REQUIRES: riscv-registered-target
; RUN: opt -p 'lto<O3>' -mtriple riscv64 -mattr=+v -S < %s | FileCheck %s

; TODO: This crashes with Assertion failed: (HeaderFreq >= BBFreq && "Header has smaller block freq than dominated BB?")

define void @f(i1 %0) !prof !0 {
  br label %2

2:                                                ; preds = %9, %1
  %3 = phi i64 [ %10, %9 ], [ 0, %1 ]
  %4 = getelementptr i64, ptr null, i64 %3
  br label %5

5:                                                ; preds = %8, %2
  %6 = phi i1 [ false, %2 ], [ true, %8 ]
  br i1 %0, label %8, label %7

7:                                                ; preds = %5
  store i64 0, ptr %4, align 8
  br label %8

8:                                                ; preds = %7, %5
  br i1 %6, label %9, label %5

9:                                                ; preds = %8
  %10 = add i64 %3, 1
  %11 = icmp eq i64 %3, 64
  br i1 %11, label %12, label %2

12:                                               ; preds = %9
  ret void
}

!0 = !{!"function_entry_count", i64 1}
