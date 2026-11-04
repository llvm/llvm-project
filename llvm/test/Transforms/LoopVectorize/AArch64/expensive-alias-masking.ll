; RUN: opt -S -disable-output -mattr=+sve2 -passes=loop-vectorize -force-partial-aliasing-vectorization -tail-folding-policy=must-fold-tail %s -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>%t
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-ALIAS-MASKING-REMARKS
; RUN: opt -S -disable-output -mattr=+sve2 -passes=loop-vectorize -tail-folding-policy=must-fold-tail %s -pass-remarks=loop-vectorize -pass-remarks-missed=loop-vectorize 2>%t
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-DIFF-CHECKS-REMARKS

target triple = "aarch64-unknown-linux-gnu"

; This loop has four store pointers and eight load pointers. It requires 38
; diff checks (no store alias = (4*3)/2 = 6 checks, no store alias with load =
; 4 * 8 = 32 checks, for a total of 38 checks). The loops trip count is low (33).
;
; Diff checks are determined to unprofitable due to the high number of checks
; and low trip count.
;
; With alias-masking we do vectorize this loop as the cost of setting up the
; alias-mask is not factored into the vectorization cost.
;
; TODO: Cost the alias-mask when using -force-partial-aliasing-vectorization.

; CHECK-DIFF-CHECKS-REMARKS: loop not vectorized

; CHECK-ALIAS-MASKING-REMARKS: vectorized loop (vectorization width: vscale x 4, interleaved count: 1)

define void @expensive_runtime_checks(ptr %0, ptr %1, ptr %2) {
entry:
  %5 = load ptr, ptr %1, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = load ptr, ptr %0, align 8
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load ptr, ptr %8, align 8
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %13 = load ptr, ptr %12, align 8
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %15 = load ptr, ptr %14, align 8
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %17 = load ptr, ptr %16, align 8
  %18 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %19 = load ptr, ptr %18, align 8
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %21 = load ptr, ptr %20, align 8
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %23 = load ptr, ptr %22, align 8
  %24 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %25 = load ptr, ptr %24, align 8
  br label %loop

loop:
  %27 = phi i64 [ 0, %entry ], [ %52, %loop ]
  %28 = getelementptr inbounds nuw [4 x i8], ptr %5, i64 %27
  %29 = load i32, ptr %28, align 4
  %30 = getelementptr inbounds nuw [4 x i8], ptr %6, i64 %27
  %31 = load i32, ptr %30, align 4
  %32 = add nsw i32 %31, %29
  %33 = getelementptr inbounds nuw [4 x i8], ptr %7, i64 %27
  store i32 %32, ptr %33, align 4
  %34 = getelementptr inbounds nuw [4 x i8], ptr %9, i64 %27
  %35 = load i32, ptr %34, align 4
  %36 = getelementptr inbounds nuw [4 x i8], ptr %11, i64 %27
  %37 = load i32, ptr %36, align 4
  %38 = add nsw i32 %37, %35
  %39 = getelementptr inbounds nuw [4 x i8], ptr %13, i64 %27
  store i32 %38, ptr %39, align 4
  %40 = getelementptr inbounds nuw [4 x i8], ptr %15, i64 %27
  %41 = load i32, ptr %40, align 4
  %42 = getelementptr inbounds nuw [4 x i8], ptr %17, i64 %27
  %43 = load i32, ptr %42, align 4
  %44 = add nsw i32 %43, %41
  %45 = getelementptr inbounds nuw [4 x i8], ptr %19, i64 %27
  store i32 %44, ptr %45, align 4
  %46 = getelementptr inbounds nuw [4 x i8], ptr %21, i64 %27
  %47 = load i32, ptr %46, align 4
  %48 = getelementptr inbounds nuw [4 x i8], ptr %23, i64 %27
  %49 = load i32, ptr %48, align 4
  %50 = add nsw i32 %49, %47
  %51 = getelementptr inbounds nuw [4 x i8], ptr %25, i64 %27
  store i32 %50, ptr %51, align 4
  %52 = add nuw nsw i64 %27, 1
  %53 = icmp eq i64 %52, 33
  br i1 %53, label %exit, label %loop

exit:
  ret void
}
