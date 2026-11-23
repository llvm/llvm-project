; RUN: opt -mattr=+simd128 -passes=loop-vectorize %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s

target triple = "wasm32"
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"

%struct.Output32x2 = type { i32, i32 }
%struct.Input8x2 = type { i8, i8 }
%struct.Output32x4 = type { i32, i32, i32, i32 }
%struct.Input8x4 = type { i8, i8, i8, i8 }
%struct.Input16x2 = type { i16, i16 }
%struct.Input16x4 = type { i16, i16, i16, i16 }
%struct.Input32x2 = type { i32, i32 }
%struct.Input32x4 = type { i32, i32, i32, i32 }

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define hidden void @accumulate8x2(ptr dead_on_unwind noalias writable sret(%struct.Output32x2) align 4 captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
; CHECK-LABEL: accumulate8x2:
; CHECK: loop
; CHECK: v128.load64_zero
; CHECK: i8x16.shuffle 1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
  %4 = load i32, ptr %0, align 4
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %10, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i32 4
  %8 = load i32, ptr %7, align 4
  br label %12

9:                                                ; preds = %12
  store i32 %23, ptr %7, align 4
  br label %10

10:                                               ; preds = %9, %3
  %11 = phi i32 [ %21, %9 ], [ %4, %3 ]
  store i32 %11, ptr %0, align 4
  ret void

12:                                               ; preds = %6, %12
  %13 = phi i32 [ %8, %6 ], [ %23, %12 ]
  %14 = phi i32 [ 0, %6 ], [ %24, %12 ]
  %15 = phi i32 [ %4, %6 ], [ %21, %12 ]
  %16 = getelementptr inbounds nuw %struct.Input8x2, ptr %1, i32 %14
  %17 = load i8, ptr %16, align 1
  %18 = getelementptr inbounds nuw i8, ptr %16, i32 1
  %19 = load i8, ptr %18, align 1
  %20 = zext i8 %17 to i32
  %21 = add i32 %15, %20
  %22 = zext i8 %19 to i32
  %23 = add i32 %13, %22
  %24 = add nuw i32 %14, 1
  %25 = icmp eq i32 %24, %2
  br i1 %25, label %9, label %12
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define hidden void @accumulate8x4(ptr dead_on_unwind noalias writable sret(%struct.Output32x4) align 4 captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
; CHECK-LABEL: accumulate8x4
; CHECK: loop
; CHECK: v128.load
; CHECK: i8x16.shuffle 3, 7, 11, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 2, 6, 10, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_u
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
  %4 = load i32, ptr %0, align 4
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %14, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i32 4
  %8 = getelementptr inbounds nuw i8, ptr %0, i32 8
  %9 = getelementptr inbounds nuw i8, ptr %0, i32 12
  %10 = load i32, ptr %7, align 4
  %11 = load i32, ptr %8, align 4
  %12 = load i32, ptr %9, align 4
  br label %16

13:                                               ; preds = %16
  store i32 %33, ptr %7, align 4
  store i32 %35, ptr %8, align 4
  store i32 %37, ptr %9, align 4
  br label %14

14:                                               ; preds = %13, %3
  %15 = phi i32 [ %31, %13 ], [ %4, %3 ]
  store i32 %15, ptr %0, align 4
  ret void

16:                                               ; preds = %6, %16
  %17 = phi i32 [ %12, %6 ], [ %37, %16 ]
  %18 = phi i32 [ %11, %6 ], [ %35, %16 ]
  %19 = phi i32 [ %10, %6 ], [ %33, %16 ]
  %20 = phi i32 [ 0, %6 ], [ %38, %16 ]
  %21 = phi i32 [ %4, %6 ], [ %31, %16 ]
  %22 = getelementptr inbounds nuw %struct.Input8x4, ptr %1, i32 %20
  %23 = load i8, ptr %22, align 1
  %24 = getelementptr inbounds nuw i8, ptr %22, i32 1
  %25 = load i8, ptr %24, align 1
  %26 = getelementptr inbounds nuw i8, ptr %22, i32 2
  %27 = load i8, ptr %26, align 1
  %28 = getelementptr inbounds nuw i8, ptr %22, i32 3
  %29 = load i8, ptr %28, align 1
  %30 = zext i8 %23 to i32
  %31 = add i32 %21, %30
  %32 = zext i8 %25 to i32
  %33 = add i32 %19, %32
  %34 = zext i8 %27 to i32
  %35 = add i32 %18, %34
  %36 = zext i8 %29 to i32
  %37 = add i32 %17, %36
  %38 = add nuw i32 %20, 1
  %39 = icmp eq i32 %38, %2
  br i1 %39, label %13, label %16
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define hidden void @accumulate16x2(ptr dead_on_unwind noalias writable sret(%struct.Output32x2) align 4 captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
; CHECK-LABEL: accumulate16x2
; CHECK: loop
; CHECK: v128.load
; CHECK: i8x16.shuffle 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
  %4 = load i32, ptr %0, align 4
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %10, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i32 4
  %8 = load i32, ptr %7, align 4
  br label %12

9:                                                ; preds = %12
  store i32 %23, ptr %7, align 4
  br label %10

10:                                               ; preds = %9, %3
  %11 = phi i32 [ %21, %9 ], [ %4, %3 ]
  store i32 %11, ptr %0, align 4
  ret void

12:                                               ; preds = %6, %12
  %13 = phi i32 [ %8, %6 ], [ %23, %12 ]
  %14 = phi i32 [ 0, %6 ], [ %24, %12 ]
  %15 = phi i32 [ %4, %6 ], [ %21, %12 ]
  %16 = getelementptr inbounds nuw %struct.Input16x2, ptr %1, i32 %14
  %17 = load i16, ptr %16, align 2
  %18 = getelementptr inbounds nuw i8, ptr %16, i32 2
  %19 = load i16, ptr %18, align 2
  %20 = zext i16 %17 to i32
  %21 = add i32 %15, %20
  %22 = zext i16 %19 to i32
  %23 = add i32 %13, %22
  %24 = add nuw i32 %14, 1
  %25 = icmp eq i32 %24, %2
  br i1 %25, label %9, label %12
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define hidden void @accumulate16x4(ptr dead_on_unwind noalias writable sret(%struct.Output32x4) align 4 captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
; CHECK-LABEL: accumulate16x4
; CHECK: loop
; CHECK: v128.load 0:p2align=1
; CHECK: v128.load 16:p2align=1
; CHECK: i8x16.shuffle 6, 7, 14, 15, 22, 23, 30, 31, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 2, 3, 10, 11, 18, 19, 26, 27, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extend_low_i16x8_u
; CHECK: i32x4.add
  %4 = load i32, ptr %0, align 4
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %14, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i32 4
  %8 = getelementptr inbounds nuw i8, ptr %0, i32 8
  %9 = getelementptr inbounds nuw i8, ptr %0, i32 12
  %10 = load i32, ptr %7, align 4
  %11 = load i32, ptr %8, align 4
  %12 = load i32, ptr %9, align 4
  br label %16

13:                                               ; preds = %16
  store i32 %33, ptr %7, align 4
  store i32 %35, ptr %8, align 4
  store i32 %37, ptr %9, align 4
  br label %14

14:                                               ; preds = %13, %3
  %15 = phi i32 [ %31, %13 ], [ %4, %3 ]
  store i32 %15, ptr %0, align 4
  ret void

16:                                               ; preds = %6, %16
  %17 = phi i32 [ %12, %6 ], [ %37, %16 ]
  %18 = phi i32 [ %11, %6 ], [ %35, %16 ]
  %19 = phi i32 [ %10, %6 ], [ %33, %16 ]
  %20 = phi i32 [ 0, %6 ], [ %38, %16 ]
  %21 = phi i32 [ %4, %6 ], [ %31, %16 ]
  %22 = getelementptr inbounds nuw %struct.Input16x4, ptr %1, i32 %20
  %23 = load i16, ptr %22, align 2
  %24 = getelementptr inbounds nuw i8, ptr %22, i32 2
  %25 = load i16, ptr %24, align 2
  %26 = getelementptr inbounds nuw i8, ptr %22, i32 4
  %27 = load i16, ptr %26, align 2
  %28 = getelementptr inbounds nuw i8, ptr %22, i32 6
  %29 = load i16, ptr %28, align 2
  %30 = zext i16 %23 to i32
  %31 = add i32 %21, %30
  %32 = zext i16 %25 to i32
  %33 = add i32 %19, %32
  %34 = zext i16 %27 to i32
  %35 = add i32 %18, %34
  %36 = zext i16 %29 to i32
  %37 = add i32 %17, %36
  %38 = add nuw i32 %20, 1
  %39 = icmp eq i32 %38, %2
  br i1 %39, label %13, label %16
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define hidden void @accumulate32x2(ptr dead_on_unwind noalias writable sret(%struct.Output32x2) align 4 captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
; CHECK-LABEL: accumulate32x2
; CHECK: loop
; CHECK: v128.load 0:p2align=2
; CHECK: v128.load 16:p2align=2
; CHECK: i8x16.shuffle 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
; CHECK: i32x4.add
  %4 = load i32, ptr %0, align 4
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %10, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i32 4
  %8 = load i32, ptr %7, align 4
  br label %12

9:                                                ; preds = %12
  store i32 %21, ptr %7, align 4
  br label %10

10:                                               ; preds = %9, %3
  %11 = phi i32 [ %20, %9 ], [ %4, %3 ]
  store i32 %11, ptr %0, align 4
  ret void

12:                                               ; preds = %6, %12
  %13 = phi i32 [ %8, %6 ], [ %21, %12 ]
  %14 = phi i32 [ 0, %6 ], [ %22, %12 ]
  %15 = phi i32 [ %4, %6 ], [ %20, %12 ]
  %16 = getelementptr inbounds nuw %struct.Input32x2, ptr %1, i32 %14
  %17 = load i32, ptr %16, align 4
  %18 = getelementptr inbounds nuw i8, ptr %16, i32 4
  %19 = load i32, ptr %18, align 4
  %20 = add i32 %15, %17
  %21 = add i32 %13, %19
  %22 = add nuw i32 %14, 1
  %23 = icmp eq i32 %22, %2
  br i1 %23, label %9, label %12
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define hidden void @accumulate32x4(ptr dead_on_unwind noalias writable sret(%struct.Output32x4) align 4 captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2) local_unnamed_addr #0 {
; CHECK-LABEL: accumulate32x4
; CHECK: v128.load 0:p2align=2
; CHECK: v128.load 16:p2align=2
; CHECK: i8x16.shuffle 12, 13, 14, 15, 28, 29, 30, 31, 0, 1, 2, 3, 0, 1, 2, 3
; CHECK: v128.load 32:p2align=2
; CHECK: v128.load 48:p2align=2
; CHECK: i8x16.shuffle 0, 1, 2, 3, 0, 1, 2, 3, 12, 13, 14, 15, 28, 29, 30, 31
; CHECK: i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 8, 9, 10, 11, 24, 25, 26, 27, 0, 1, 2, 3, 0, 1, 2, 3
; CHECK: i8x16.shuffle 0, 1, 2, 3, 0, 1, 2, 3, 8, 9, 10, 11, 24, 25, 26, 27
; CHECK: i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 4, 5, 6, 7, 20, 21, 22, 23, 0, 1, 2, 3, 0, 1, 2, 3
; CHECK: i8x16.shuffle 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23
; CHECK: i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; CHECK: i32x4.add
; CHECK: i8x16.shuffle 0, 1, 2, 3, 16, 17, 18, 19, 0, 1, 2, 3, 0, 1, 2, 3
; CHECK: i8x16.shuffle 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 16, 17, 18, 19
; CHECK: i8x16.shuffle 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; CHECK: i32x4.add
  %4 = load i32, ptr %0, align 4
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %14, label %6

6:                                                ; preds = %3
  %7 = getelementptr inbounds nuw i8, ptr %0, i32 4
  %8 = getelementptr inbounds nuw i8, ptr %0, i32 8
  %9 = getelementptr inbounds nuw i8, ptr %0, i32 12
  %10 = load i32, ptr %7, align 4
  %11 = load i32, ptr %8, align 4
  %12 = load i32, ptr %9, align 4
  br label %16

13:                                               ; preds = %16
  store i32 %31, ptr %7, align 4
  store i32 %32, ptr %8, align 4
  store i32 %33, ptr %9, align 4
  br label %14

14:                                               ; preds = %13, %3
  %15 = phi i32 [ %30, %13 ], [ %4, %3 ]
  store i32 %15, ptr %0, align 4
  ret void

16:                                               ; preds = %6, %16
  %17 = phi i32 [ %12, %6 ], [ %33, %16 ]
  %18 = phi i32 [ %11, %6 ], [ %32, %16 ]
  %19 = phi i32 [ %10, %6 ], [ %31, %16 ]
  %20 = phi i32 [ 0, %6 ], [ %34, %16 ]
  %21 = phi i32 [ %4, %6 ], [ %30, %16 ]
  %22 = getelementptr inbounds nuw %struct.Input32x4, ptr %1, i32 %20
  %23 = load i32, ptr %22, align 4
  %24 = getelementptr inbounds nuw i8, ptr %22, i32 4
  %25 = load i32, ptr %24, align 4
  %26 = getelementptr inbounds nuw i8, ptr %22, i32 8
  %27 = load i32, ptr %26, align 4
  %28 = getelementptr inbounds nuw i8, ptr %22, i32 12
  %29 = load i32, ptr %28, align 4
  %30 = add i32 %21, %23
  %31 = add i32 %19, %25
  %32 = add i32 %18, %27
  %33 = add i32 %17, %29
  %34 = add nuw i32 %20, 1
  %35 = icmp eq i32 %34, %2
  br i1 %35, label %13, label %16
}
