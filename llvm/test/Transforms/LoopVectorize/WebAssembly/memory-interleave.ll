; REQUIRES: asserts
; RUN: opt -mattr=+simd128 -passes=loop-vectorize -debug-only=loop-vectorize,vectorutils -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-wasi"

%struct.TwoInts = type { i32, i32 }
%struct.ThreeInts = type { i32, i32, i32 }
%struct.FourInts = type { i32, i32, i32, i32 }
%struct.ThreeShorts = type { i16, i16, i16 }
%struct.FourShorts = type { i16, i16, i16, i16 }
%struct.TwoBytes = type { i8, i8 }
%struct.ThreeBytes = type { i8, i8, i8 }
%struct.FourBytes = type { i8, i8, i8, i8 }
%struct.FiveBytes = type { i8, i8, i8, i8, i8 }
%struct.EightBytes = type { i8, i8, i8, i8, i8, i8, i8, i8 }

; CHECK-LABEL: two_ints_same_op
; CHECK: Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 2 at %10
; CHECK: Cost of 6 for VF 4: INTERLEAVE-GROUP with factor 2 at %10
; CHECK: LV: Scalar loop costs: 12.
; CHECK: LV: Vector loop of width 2 costs: 13.
; CHECK: LV: Vector loop of width 4 costs: 6.
; CHECK: LV: Selecting VF: 4
define hidden void @two_ints_same_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %21, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.TwoInts, ptr %1, i32 %8
  %10 = load i32, ptr %9, align 4
  %11 = getelementptr inbounds %struct.TwoInts, ptr %2, i32 %8
  %12 = load i32, ptr %11, align 4
  %13 = add i32 %12, %10
  %14 = getelementptr inbounds %struct.TwoInts, ptr %0, i32 %8
  store i32 %13, ptr %14, align 4
  %15 = getelementptr inbounds i8, ptr %9, i32 4
  %16 = load i32, ptr %15, align 4
  %17 = getelementptr inbounds i8, ptr %11, i32 4
  %18 = load i32, ptr %17, align 4
  %19 = add i32 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 4
  store i32 %19, ptr %20, align 4
  %21 = add nuw i32 %8, 1
  %22 = icmp eq i32 %21, %3
  br i1 %22, label %6, label %7
}

; CHECK-LABEL: two_ints_vary_op
; CHECK: Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 2 at %10
; CHECK: Cost of 6 for VF 4: INTERLEAVE-GROUP with factor 2 at %10
; CHECK: LV: Scalar loop costs: 12.
; CHECK: LV: Vector loop of width 2 costs: 13.
; CHECK: LV: Vector loop of width 4 costs: 6.
; CHECK: LV: Selecting VF: 4
define hidden void @two_ints_vary_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %21, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.TwoInts, ptr %1, i32 %8
  %10 = load i32, ptr %9, align 4
  %11 = getelementptr inbounds %struct.TwoInts, ptr %2, i32 %8
  %12 = load i32, ptr %11, align 4
  %13 = add i32 %12, %10
  %14 = getelementptr inbounds %struct.TwoInts, ptr %0, i32 %8
  store i32 %13, ptr %14, align 4
  %15 = getelementptr inbounds i8, ptr %9, i32 4
  %16 = load i32, ptr %15, align 4
  %17 = getelementptr inbounds i8, ptr %11, i32 4
  %18 = load i32, ptr %17, align 4
  %19 = sub i32 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 4
  store i32 %19, ptr %20, align 4
  %21 = add nuw i32 %8, 1
  %22 = icmp eq i32 %21, %3
  br i1 %22, label %6, label %7
}

; CHECK-LABEL: three_ints
; CHECK: Cost of 14 for VF 2: INTERLEAVE-GROUP with factor 3 at
; CHECK: Cost of 28 for VF 4: INTERLEAVE-GROUP with factor 3 at
; CHECK: LV: Scalar loop costs: 16.
; CHECK: LV: Found an estimated cost of 14 for VF 2 For instruction: %10 = load i32, ptr %9
; CHECK: LV: Found an estimated cost of 14 for VF 2 For instruction: %12 = load i32, ptr %11
; CHECK: LV: Found an estimated cost of 14 for VF 2 For instruction: store i32 %25, ptr %26
; CHECK: LV: Vector loop of width 2 costs: 24.
; CHECK: LV: Found an estimated cost of 28 for VF 4 For instruction: %10 = load i32, ptr %9
; CHECK: LV: Found an estimated cost of 28 for VF 4 For instruction: %12 = load i32, ptr %11
; CHECK: LV: Found an estimated cost of 28 for VF 4 For instruction: store i32 %25, ptr %26
; CHECK: LV: Vector loop of width 4 costs: 22.
; CHECK: LV: Selecting VF: 1
define hidden void @three_ints(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %27, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.ThreeInts, ptr %1, i32 %8
  %10 = load i32, ptr %9, align 4
  %11 = getelementptr inbounds %struct.ThreeInts, ptr %2, i32 %8
  %12 = load i32, ptr %11, align 4
  %13 = add nsw i32 %12, %10
  %14 = getelementptr inbounds %struct.ThreeInts, ptr %0, i32 %8
  store i32 %13, ptr %14, align 4
  %15 = getelementptr inbounds i8, ptr %9, i32 4
  %16 = load i32, ptr %15, align 4
  %17 = getelementptr inbounds i8, ptr %11, i32 4
  %18 = load i32, ptr %17, align 4
  %19 = add nsw i32 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 4
  store i32 %19, ptr %20, align 4
  %21 = getelementptr inbounds i8, ptr %9, i32 8
  %22 = load i32, ptr %21, align 4
  %23 = getelementptr inbounds i8, ptr %11, i32 8
  %24 = load i32, ptr %23, align 4
  %25 = add nsw i32 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 8
  store i32 %25, ptr %26, align 4
  %27 = add nuw i32 %8, 1
  %28 = icmp eq i32 %27, %3
  br i1 %28, label %6, label %7
}

; CHECK-LABEL: three_shorts
; CHECK: Cost of 26 for VF 4: INTERLEAVE-GROUP with factor 3
; CHECK: Cost of 52 for VF 8: INTERLEAVE-GROUP with factor 3
; CHECK: LV: Scalar loop costs: 16.
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: store i16 %25
; CHECK: LV: Vector loop of width 2 costs: 30.
; CHECK: LV: Found an estimated cost of 26 for VF 4 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 26 for VF 4 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 26 for VF 4 For instruction: store i16 %25
; CHECK: LV: Vector loop of width 4 costs: 21.
; CHECK: LV: Found an estimated cost of 52 for VF 8 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 52 for VF 8 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 52 for VF 8 For instruction: store i16 %25
; CHECK: LV: Vector loop of width 8 costs: 20.
; CHECK: LV: Selecting VF: 1
define hidden void @three_shorts(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %27, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.ThreeShorts, ptr %1, i32 %8
  %10 = load i16, ptr %9, align 2
  %11 = getelementptr inbounds %struct.ThreeShorts, ptr %2, i32 %8
  %12 = load i16, ptr %11, align 2
  %13 = mul i16 %12, %10
  %14 = getelementptr inbounds %struct.ThreeShorts, ptr %0, i32 %8
  store i16 %13, ptr %14, align 2
  %15 = getelementptr inbounds i8, ptr %9, i32 2
  %16 = load i16, ptr %15, align 2
  %17 = getelementptr inbounds i8, ptr %11, i32 2
  %18 = load i16, ptr %17, align 2
  %19 = mul i16 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 2
  store i16 %19, ptr %20, align 2
  %21 = getelementptr inbounds i8, ptr %9, i32 4
  %22 = load i16, ptr %21, align 2
  %23 = getelementptr inbounds i8, ptr %11, i32 4
  %24 = load i16, ptr %23, align 2
  %25 = mul i16 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 4
  store i16 %25, ptr %26, align 2
  %27 = add nuw i32 %8, 1
  %28 = icmp eq i32 %27, %3
  br i1 %28, label %6, label %7
}

; CHECK-LABEL: four_shorts_same_op
; CHECK: Cost of 18 for VF 2: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 68 for VF 8: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 20.
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: store i16
; CHECK: LV: Vector loop of width 2 costs: 31.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: store i16
; CHECK: LV: Vector loop of width 4 costs: 15.
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: store i16
; CHECK: LV: Vector loop of width 8 costs: 26
; CHECK: LV: Selecting VF: 4
define hidden void @four_shorts_same_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %33, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourShorts, ptr %1, i32 %8
  %10 = load i16, ptr %9, align 2
  %11 = getelementptr inbounds %struct.FourShorts, ptr %2, i32 %8
  %12 = load i16, ptr %11, align 2
  %13 = sub i16 %10, %12
  %14 = getelementptr inbounds %struct.FourShorts, ptr %0, i32 %8
  store i16 %13, ptr %14, align 2
  %15 = getelementptr inbounds i8, ptr %9, i32 2
  %16 = load i16, ptr %15, align 2
  %17 = getelementptr inbounds i8, ptr %11, i32 2
  %18 = load i16, ptr %17, align 2
  %19 = sub i16 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 2
  store i16 %19, ptr %20, align 2
  %21 = getelementptr inbounds i8, ptr %9, i32 4
  %22 = load i16, ptr %21, align 2
  %23 = getelementptr inbounds i8, ptr %11, i32 4
  %24 = load i16, ptr %23, align 2
  %25 = sub i16 %22, %24
  %26 = getelementptr inbounds i8, ptr %14, i32 4
  store i16 %25, ptr %26, align 2
  %27 = getelementptr inbounds i8, ptr %9, i32 6
  %28 = load i16, ptr %27, align 2
  %29 = getelementptr inbounds i8, ptr %11, i32 6
  %30 = load i16, ptr %29, align 2
  %31 = sub i16 %28, %30
  %32 = getelementptr inbounds i8, ptr %14, i32 6
  store i16 %31, ptr %32, align 2
  %33 = add nuw i32 %8, 1
  %34 = icmp eq i32 %33, %3
  br i1 %34, label %6, label %7
}

; CHECK-LABEL: four_shorts_split_op
; CHECK: Cost of 18 for VF 2: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 68 for VF 8: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 20.
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: store i16
; CHECK: LV: Vector loop of width 2 costs: 31.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: store i16 %31
; CHECK: LV: Vector loop of width 4 costs: 15.
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: store i16 %31
; CHECK: LV: Vector loop of width 8 costs: 26.
; CHECK: LV: Selecting VF: 4
define hidden void @four_shorts_split_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %33, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourShorts, ptr %1, i32 %8
  %10 = load i16, ptr %9, align 2
  %11 = getelementptr inbounds %struct.FourShorts, ptr %2, i32 %8
  %12 = load i16, ptr %11, align 2
  %13 = or i16 %12, %10
  %14 = getelementptr inbounds %struct.FourShorts, ptr %0, i32 %8
  store i16 %13, ptr %14, align 2
  %15 = getelementptr inbounds i8, ptr %9, i32 2
  %16 = load i16, ptr %15, align 2
  %17 = getelementptr inbounds i8, ptr %11, i32 2
  %18 = load i16, ptr %17, align 2
  %19 = or i16 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 2
  store i16 %19, ptr %20, align 2
  %21 = getelementptr inbounds i8, ptr %9, i32 4
  %22 = load i16, ptr %21, align 2
  %23 = getelementptr inbounds i8, ptr %11, i32 4
  %24 = load i16, ptr %23, align 2
  %25 = xor i16 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 4
  store i16 %25, ptr %26, align 2
  %27 = getelementptr inbounds i8, ptr %9, i32 6
  %28 = load i16, ptr %27, align 2
  %29 = getelementptr inbounds i8, ptr %11, i32 6
  %30 = load i16, ptr %29, align 2
  %31 = xor i16 %30, %28
  %32 = getelementptr inbounds i8, ptr %14, i32 6
  store i16 %31, ptr %32, align 2
  %33 = add nuw i32 %8, 1
  %34 = icmp eq i32 %33, %3
  br i1 %34, label %6, label %7
}

; CHECK-LABEL: four_shorts_interleave_op
; CHECK: Cost of 18 for VF 2: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 68 for VF 8: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 20.
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 2 For instruction: store i16
; CHECK: LV: Vector loop of width 2 costs: 31.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: store i16
; CHECK: LV: Vector loop of width 4 costs: 15.
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: %10 = load i16
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: %12 = load i16
; CHECK: LV: Found an estimated cost of 68 for VF 8 For instruction: store i16
; CHECK: LV: Vector loop of width 8 costs: 26.
; CHECK: LV: Selecting VF: 4
define hidden void @four_shorts_interleave_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %33, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourShorts, ptr %1, i32 %8
  %10 = load i16, ptr %9, align 2
  %11 = getelementptr inbounds %struct.FourShorts, ptr %2, i32 %8
  %12 = load i16, ptr %11, align 2
  %13 = or i16 %12, %10
  %14 = getelementptr inbounds %struct.FourShorts, ptr %0, i32 %8
  store i16 %13, ptr %14, align 2
  %15 = getelementptr inbounds i8, ptr %9, i32 2
  %16 = load i16, ptr %15, align 2
  %17 = getelementptr inbounds i8, ptr %11, i32 2
  %18 = load i16, ptr %17, align 2
  %19 = xor i16 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 2
  store i16 %19, ptr %20, align 2
  %21 = getelementptr inbounds i8, ptr %9, i32 4
  %22 = load i16, ptr %21, align 2
  %23 = getelementptr inbounds i8, ptr %11, i32 4
  %24 = load i16, ptr %23, align 2
  %25 = or i16 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 4
  store i16 %25, ptr %26, align 2
  %27 = getelementptr inbounds i8, ptr %9, i32 6
  %28 = load i16, ptr %27, align 2
  %29 = getelementptr inbounds i8, ptr %11, i32 6
  %30 = load i16, ptr %29, align 2
  %31 = xor i16 %30, %28
  %32 = getelementptr inbounds i8, ptr %14, i32 6
  store i16 %31, ptr %32, align 2
  %33 = add nuw i32 %8, 1
  %34 = icmp eq i32 %33, %3
  br i1 %34, label %6, label %7
}

; CHECK-LABEL: five_shorts
; CHECK: Cost of 42 for VF 4: INTERLEAVE-GROUP with factor 5
; CHECK: Cost of 84 for VF 8: INTERLEAVE-GROUP with factor 5
; CHECK: LV: Found an estimated cost of 84 for VF 8 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 84 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 84 for VF 8 For instruction: store i8 %37
; CHECK: LV: Vector loop of width 8 costs: 32
; CHECK: LV: Found an estimated cost of 168 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 168 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 168 for VF 16 For instruction: store i8 %37
; CHECK: LV: Vector loop of width 16 costs: 32
; CHECK: LV: Selecting VF: 1
define hidden void @five_shorts(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %39, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FiveBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.FiveBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = sub i8 %10, %12
  %14 = getelementptr inbounds %struct.FiveBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = sub i8 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = sub i8 %22, %24
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = sub i8 %28, %30
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = getelementptr inbounds i8, ptr %9, i32 4
  %34 = load i8, ptr %33, align 1
  %35 = getelementptr inbounds i8, ptr %11, i32 4
  %36 = load i8, ptr %35, align 1
  %37 = sub i8 %34, %36
  %38 = getelementptr inbounds i8, ptr %14, i32 4
  store i8 %37, ptr %38, align 1
  %39 = add nuw i32 %8, 1
  %40 = icmp eq i32 %39, %3
  br i1 %40, label %6, label %7
}

; CHECK-LABEL: two_bytes_same_op
; CHECK: Cost of 11 for VF 4: INTERLEAVE-GROUP with factor 2
; CHECK: Cost of 7 for VF 8: INTERLEAVE-GROUP with factor 2
; CHECK: Cost of 6 for VF 16: INTERLEAVE-GROUP with factor 2
; CHECK: LV: Scalar loop costs: 12.
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: store i8 %13
; CHECK: LV: Vector loop of width 2 costs: 26.
; CHECK: LV: Found an estimated cost of 11 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 12 for VF 4 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 11 for VF 4 For instruction: store i8
; CHECK: LV: Vector loop of width 4 costs: 15.
; CHECK: LV: Found an estimated cost of 7 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 24 for VF 8 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 7 for VF 8 For instruction: store i8
; CHECK: LV: Vector loop of width 8 costs: 9.
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 48 for VF 16 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction: store i8
; CHECK: LV: Vector loop of width 16 costs: 7.
; CHECK: LV: Selecting VF: 16.
define hidden void @two_bytes_same_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %21, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.TwoBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.TwoBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = mul i8 %12, %10
  %14 = getelementptr inbounds %struct.TwoBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = mul i8 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = add nuw i32 %8, 1
  %22 = icmp eq i32 %21, %3
  br i1 %22, label %6, label %7
}

; CHECK-LABEL: two_bytes_vary_op
; CHECK: Cost of 11 for VF 4: INTERLEAVE-GROUP with factor 2
; CHECK: Cost of 7 for VF 8: INTERLEAVE-GROUP with factor 2
; CHECK: Cost of 6 for VF 16: INTERLEAVE-GROUP with factor 2
; CHECK: LV: Scalar loop costs: 12.
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: store i8 %13
; CHECK: LV: Vector loop of width 2 costs: 23.
; CHECK: LV: Found an estimated cost of 11 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 12 for VF 4 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 11 for VF 4 For instruction: store i8
; CHECK: LV: Vector loop of width 4 costs: 12.
; CHECK: LV: Found an estimated cost of 7 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 24 for VF 8 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 7 for VF 8 For instruction: store i8
; CHECK: LV: Vector loop of width 8 costs: 6.
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 48 for VF 16 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction: store i8 %19
; CHECK: LV: Vector loop of width 16 costs: 4.
; CHECK: LV: Selecting VF: 16.
define hidden void @two_bytes_vary_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %21, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.TwoBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.TwoBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = mul i8 %12, %10
  %14 = getelementptr inbounds %struct.TwoBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = sub i8 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = add nuw i32 %8, 1
  %22 = icmp eq i32 %21, %3
  br i1 %22, label %6, label %7
}

; CHECK-LABEL: three_bytes_same_op
; CHECK: Cost of 50 for VF 8: INTERLEAVE-GROUP with factor 3 at %10
; CHECK: Cost of 100 for VF 16: INTERLEAVE-GROUP with factor 3 at %10
; CHECK: LV: Scalar loop costs: 16.
; CHECK: LV: Vector loop of width 2 costs: 30.
; CHECK: LV: Vector loop of width 4 costs: 28.
; CHECK: LV: Found an estimated cost of 50 for VF 8 For instruction: %10 = load i8, ptr %9
; CHECK: LV: Found an estimated cost of 50 for VF 8 For instruction: %12 = load i8, ptr %11
; CHECK: LV: Found an estimated cost of 50 for VF 8 For instruction: store i8 %25
; CHECK: LV: Vector loop of width 8 costs: 19.
; CHECK: LV: Found an estimated cost of 100 for VF 16 For instruction: %10 = load i8, ptr %9
; CHECK: LV: Found an estimated cost of 100 for VF 16 For instruction: %12 = load i8, ptr %11
; CHECK: LV: Found an estimated cost of 100 for VF 16 For instruction: store i8 %25
; CHECK: LV: Vector loop of width 16 costs: 19.
; CHECK: LV: Selecting VF: 1.
define hidden void @three_bytes_same_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %27, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.ThreeBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.ThreeBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = and i8 %12, %10
  %14 = getelementptr inbounds %struct.ThreeBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = and i8 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = and i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = add nuw i32 %8, 1
  %28 = icmp eq i32 %27, %3
  br i1 %28, label %6, label %7
}

; CHECK-LABEL: three_bytes_interleave_op
; CHECK: Cost of 50 for VF 8: INTERLEAVE-GROUP with factor 3 at %10, ir<%9>
; CHECK: Cost of 100 for VF 16: INTERLEAVE-GROUP with factor 3 at %10, ir<%9>
; CHECK: LV: Scalar loop costs: 16.
; CHECK: LV: Vector loop of width 2 costs: 30.
; CHECK: LV: Vector loop of width 4 costs: 28.
; CHECK: LV: Found an estimated cost of 50 for VF 8 For instruction: %10 = load i8, ptr %9
; CHECK: LV: Found an estimated cost of 50 for VF 8 For instruction: %12 = load i8, ptr %11
; CHECK: LV: Found an estimated cost of 50 for VF 8 For instruction: store i8 %25
; CHECK: LV: Vector loop of width 8 costs: 19.
; CHECK: LV: Found an estimated cost of 100 for VF 16 For instruction: %10 = load i8, ptr %9
; CHECK: LV: Found an estimated cost of 100 for VF 16 For instruction: %12 = load i8, ptr %11
; CHECK: LV: Found an estimated cost of 100 for VF 16 For instruction: store i8 %25
; CHECK: LV: Vector loop of width 16 costs: 19.
; CHECK: LV: Selecting VF: 1.
define hidden void @three_bytes_interleave_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %27, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.ThreeBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.ThreeBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = add i8 %12, %10
  %14 = getelementptr inbounds %struct.ThreeBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = sub i8 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = add i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = add nuw i32 %8, 1
  %28 = icmp eq i32 %27, %3
  br i1 %28, label %6, label %7
}

; CHECK-LABEL: four_bytes_same_op
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 26 for VF 8: INTERLEAVE-GROUP with factor 4 
; CHECK: Cost of 132 for VF 16: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 20.
; CHECK: LV: Vector loop of width 2 costs: 40.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: store i8
; CHECK: LV: Vector loop of width 4 costs: 15.
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: store i8
; CHECK: LV: Vector loop of width 8 costs: 10.
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: store i8
; CHECK: LV: Vector loop of width 16 costs: 25.
; CHECK: LV: Selecting VF: 8.
define hidden void @four_bytes_same_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %33, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.FourBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = and i8 %12, %10
  %14 = getelementptr inbounds %struct.FourBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = and i8 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = and i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = and i8 %30, %28
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = add nuw i32 %8, 1
  %34 = icmp eq i32 %33, %3
  br i1 %34, label %6, label %7
}

; CHECK-LABEL: four_bytes_split_op
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 26 for VF 8: INTERLEAVE-GROUP with factor 4 
; CHECK: Cost of 132 for VF 16: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 20.
; CHECK: LV: Vector loop of width 2 costs: 45.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 12 for VF 4 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: store i8
; CHECK: LV: Vector loop of width 4 costs: 21.
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 24 for VF 8 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: store i8
; CHECK: LV: Vector loop of width 8 costs: 16.
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 48 for VF 16 For instruction: %13 = mul i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: store i8
; CHECK: LV: Vector loop of width 16 costs: 31
; CHECK: LV: Selecting VF: 8
define hidden void @four_bytes_split_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %33, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.FourBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = mul i8 %12, %10
  %14 = getelementptr inbounds %struct.FourBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = mul i8 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = sub i8 %22, %24
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = sub i8 %28, %30
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = add nuw i32 %8, 1
  %34 = icmp eq i32 %33, %3
  br i1 %34, label %6, label %7
}


; CHECK-LABEL: four_bytes_interleave_op
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 26 for VF 8: INTERLEAVE-GROUP with factor 4 
; CHECK: Cost of 132 for VF 16: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 20.
; CHECK: LV: Vector loop of width 2 costs: 40
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: store i8
; CHECK: LV: Vector loop of width 4 costs: 15
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: store i8
; CHECK: LV: Vector loop of width 8 costs: 10
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: store i8
; CHECK: LV: Vector loop of width 16 costs: 25
; CHECK: LV: Selecting VF: 8
define hidden void @four_bytes_interleave_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %33, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.FourBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = add i8 %12, %10
  %14 = getelementptr inbounds %struct.FourBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = sub i8 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = add i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = sub i8 %28, %30
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = add nuw i32 %8, 1
  %34 = icmp eq i32 %33, %3
  br i1 %34, label %6, label %7
}


; CHECK-LABEL: eight_bytes_same_op
; CHECK: Cost of 34 for VF 2: INTERLEAVE-GROUP with factor 8
; CHECK: Cost of 66 for VF 4: INTERLEAVE-GROUP with factor 8
; CHECK: Cost of 132 for VF 8: INTERLEAVE-GROUP with factor 8
; CHECK: LV: Found an estimated cost of 66 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 66 for VF 4 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 66 for VF 4 For instruction: store i8 %55
; CHECK: LV: Vector loop of width 4 costs: 74
; CHECK: LV: Found an estimated cost of 132 for VF 8 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 8 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 8 For instruction: store i8 %55
; CHECK: LV: Vector loop of width 8 costs: 74
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: store i8 %55
; CHECK: LV: Vector loop of width 16 costs: 73
; CHECK: LV: Selecting VF: 1
define hidden void @eight_bytes_same_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %57, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.EightBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.EightBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = mul i8 %12, %10
  %14 = getelementptr inbounds %struct.EightBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = mul i8 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = mul i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = mul i8 %30, %28
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = getelementptr inbounds i8, ptr %9, i32 4
  %34 = load i8, ptr %33, align 1
  %35 = getelementptr inbounds i8, ptr %11, i32 4
  %36 = load i8, ptr %35, align 1
  %37 = mul i8 %36, %34
  %38 = getelementptr inbounds i8, ptr %14, i32 4
  store i8 %37, ptr %38, align 1
  %39 = getelementptr inbounds i8, ptr %9, i32 5
  %40 = load i8, ptr %39, align 1
  %41 = getelementptr inbounds i8, ptr %11, i32 5
  %42 = load i8, ptr %41, align 1
  %43 = mul i8 %42, %40
  %44 = getelementptr inbounds i8, ptr %14, i32 5
  store i8 %43, ptr %44, align 1
  %45 = getelementptr inbounds i8, ptr %9, i32 6
  %46 = load i8, ptr %45, align 1
  %47 = getelementptr inbounds i8, ptr %11, i32 6
  %48 = load i8, ptr %47, align 1
  %49 = mul i8 %48, %46
  %50 = getelementptr inbounds i8, ptr %14, i32 6
  store i8 %49, ptr %50, align 1
  %51 = getelementptr inbounds i8, ptr %9, i32 7
  %52 = load i8, ptr %51, align 1
  %53 = getelementptr inbounds i8, ptr %11, i32 7
  %54 = load i8, ptr %53, align 1
  %55 = mul i8 %54, %52
  %56 = getelementptr inbounds i8, ptr %14, i32 7
  store i8 %55, ptr %56, align 1
  %57 = add nuw i32 %8, 1
  %58 = icmp eq i32 %57, %3
  br i1 %58, label %6, label %7
}

; CHECK-LABEL: eight_bytes_split_op
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: store i8 %55
; CHECK: LV: Vector loop of width 16 costs: 50
; CHECK: LV: Selecting VF: 1
define hidden void @eight_bytes_split_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %57, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.EightBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.EightBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = add i8 %12, %10
  %14 = getelementptr inbounds %struct.EightBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = add i8 %18, %16
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = add i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = add i8 %30, %28
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = getelementptr inbounds i8, ptr %9, i32 4
  %34 = load i8, ptr %33, align 1
  %35 = getelementptr inbounds i8, ptr %11, i32 4
  %36 = load i8, ptr %35, align 1
  %37 = sub i8 %34, %36
  %38 = getelementptr inbounds i8, ptr %14, i32 4
  store i8 %37, ptr %38, align 1
  %39 = getelementptr inbounds i8, ptr %9, i32 5
  %40 = load i8, ptr %39, align 1
  %41 = getelementptr inbounds i8, ptr %11, i32 5
  %42 = load i8, ptr %41, align 1
  %43 = sub i8 %40, %42
  %44 = getelementptr inbounds i8, ptr %14, i32 5
  store i8 %43, ptr %44, align 1
  %45 = getelementptr inbounds i8, ptr %9, i32 6
  %46 = load i8, ptr %45, align 1
  %47 = getelementptr inbounds i8, ptr %11, i32 6
  %48 = load i8, ptr %47, align 1
  %49 = sub i8 %46, %48
  %50 = getelementptr inbounds i8, ptr %14, i32 6
  store i8 %49, ptr %50, align 1
  %51 = getelementptr inbounds i8, ptr %9, i32 7
  %52 = load i8, ptr %51, align 1
  %53 = getelementptr inbounds i8, ptr %11, i32 7
  %54 = load i8, ptr %53, align 1
  %55 = sub i8 %52, %54
  %56 = getelementptr inbounds i8, ptr %14, i32 7
  store i8 %55, ptr %56, align 1
  %57 = add nuw i32 %8, 1
  %58 = icmp eq i32 %57, %3
  br i1 %58, label %6, label %7
}

; CHECK-LABEL: eight_bytes_interleave_op
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: %12 = load i8
; CHECK: LV: Found an estimated cost of 264 for VF 16 For instruction: store i8 %55
; CHECK: LV: Vector loop of width 16 costs: 50
; CHECK: LV: Selecting VF: 1
define hidden void @eight_bytes_interleave_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %57, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.EightBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = getelementptr inbounds %struct.EightBytes, ptr %2, i32 %8
  %12 = load i8, ptr %11, align 1
  %13 = add i8 %12, %10
  %14 = getelementptr inbounds %struct.EightBytes, ptr %0, i32 %8
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %9, i32 1
  %16 = load i8, ptr %15, align 1
  %17 = getelementptr inbounds i8, ptr %11, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = sub i8 %16, %18
  %20 = getelementptr inbounds i8, ptr %14, i32 1
  store i8 %19, ptr %20, align 1
  %21 = getelementptr inbounds i8, ptr %9, i32 2
  %22 = load i8, ptr %21, align 1
  %23 = getelementptr inbounds i8, ptr %11, i32 2
  %24 = load i8, ptr %23, align 1
  %25 = add i8 %24, %22
  %26 = getelementptr inbounds i8, ptr %14, i32 2
  store i8 %25, ptr %26, align 1
  %27 = getelementptr inbounds i8, ptr %9, i32 3
  %28 = load i8, ptr %27, align 1
  %29 = getelementptr inbounds i8, ptr %11, i32 3
  %30 = load i8, ptr %29, align 1
  %31 = sub i8 %28, %30
  %32 = getelementptr inbounds i8, ptr %14, i32 3
  store i8 %31, ptr %32, align 1
  %33 = getelementptr inbounds i8, ptr %9, i32 4
  %34 = load i8, ptr %33, align 1
  %35 = getelementptr inbounds i8, ptr %11, i32 4
  %36 = load i8, ptr %35, align 1
  %37 = add i8 %36, %34
  %38 = getelementptr inbounds i8, ptr %14, i32 4
  store i8 %37, ptr %38, align 1
  %39 = getelementptr inbounds i8, ptr %9, i32 5
  %40 = load i8, ptr %39, align 1
  %41 = getelementptr inbounds i8, ptr %11, i32 5
  %42 = load i8, ptr %41, align 1
  %43 = sub i8 %40, %42
  %44 = getelementptr inbounds i8, ptr %14, i32 5
  store i8 %43, ptr %44, align 1
  %45 = getelementptr inbounds i8, ptr %9, i32 6
  %46 = load i8, ptr %45, align 1
  %47 = getelementptr inbounds i8, ptr %11, i32 6
  %48 = load i8, ptr %47, align 1
  %49 = add i8 %48, %46
  %50 = getelementptr inbounds i8, ptr %14, i32 6
  store i8 %49, ptr %50, align 1
  %51 = getelementptr inbounds i8, ptr %9, i32 7
  %52 = load i8, ptr %51, align 1
  %53 = getelementptr inbounds i8, ptr %11, i32 7
  %54 = load i8, ptr %53, align 1
  %55 = sub i8 %52, %54
  %56 = getelementptr inbounds i8, ptr %14, i32 7
  store i8 %55, ptr %56, align 1
  %57 = add nuw i32 %8, 1
  %58 = icmp eq i32 %57, %3
  br i1 %58, label %6, label %7
}

; CHECK-LABEL: four_bytes_into_four_ints_same_op
; CHECK: LV: Scalar loop costs: 28.
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 14 for VF 2 For instruction: %17 = load i32
; CHECK: LV: Found an estimated cost of 14 for VF 2 For instruction: store i32
; CHECK: LV: Vector loop of width 2 costs: 44.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 24 for VF 4 For instruction: %17 = load i32
; CHECK: LV: Found an estimated cost of 24 for VF 4 For instruction: store i32
; CHECK: LV: Vector loop of width 4 costs: 26.
; CHECK: LV: Selecting VF: 4.
define hidden void @four_bytes_into_four_ints_same_op(ptr noalias nocapture noundef %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %49, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = zext i8 %10 to i32
  %12 = getelementptr inbounds %struct.FourBytes, ptr %2, i32 %8
  %13 = load i8, ptr %12, align 1
  %14 = zext i8 %13 to i32
  %15 = mul nuw nsw i32 %14, %11
  %16 = getelementptr inbounds %struct.FourInts, ptr %0, i32 %8
  %17 = load i32, ptr %16, align 4
  %18 = add nsw i32 %15, %17
  store i32 %18, ptr %16, align 4
  %19 = getelementptr inbounds i8, ptr %9, i32 1
  %20 = load i8, ptr %19, align 1
  %21 = zext i8 %20 to i32
  %22 = getelementptr inbounds i8, ptr %12, i32 1
  %23 = load i8, ptr %22, align 1
  %24 = zext i8 %23 to i32
  %25 = mul nuw nsw i32 %24, %21
  %26 = getelementptr inbounds i8, ptr %16, i32 4
  %27 = load i32, ptr %26, align 4
  %28 = add nsw i32 %25, %27
  store i32 %28, ptr %26, align 4
  %29 = getelementptr inbounds i8, ptr %9, i32 2
  %30 = load i8, ptr %29, align 1
  %31 = zext i8 %30 to i32
  %32 = getelementptr inbounds i8, ptr %12, i32 2
  %33 = load i8, ptr %32, align 1
  %34 = zext i8 %33 to i32
  %35 = mul nuw nsw i32 %34, %31
  %36 = getelementptr inbounds i8, ptr %16, i32 8
  %37 = load i32, ptr %36, align 4
  %38 = add nsw i32 %35, %37
  store i32 %38, ptr %36, align 4
  %39 = getelementptr inbounds i8, ptr %9, i32 3
  %40 = load i8, ptr %39, align 1
  %41 = zext i8 %40 to i32
  %42 = getelementptr inbounds i8, ptr %12, i32 3
  %43 = load i8, ptr %42, align 1
  %44 = zext i8 %43 to i32
  %45 = mul nuw nsw i32 %44, %41
  %46 = getelementptr inbounds i8, ptr %16, i32 12
  %47 = load i32, ptr %46, align 4
  %48 = add nsw i32 %45, %47
  store i32 %48, ptr %46, align 4
  %49 = add nuw i32 %8, 1
  %50 = icmp eq i32 %49, %3
  br i1 %50, label %6, label %7
}

; CHECK-LABEL: four_bytes_into_four_ints_vary_op
; CHECK: Cost of 14 for VF 2: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 18 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 24 for VF 4: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Scalar loop costs: 21.
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 0 for VF 2 For instruction: %11 = zext i8
; CHECK: LV: Found an estimated cost of 14 for VF 2 For instruction: store i32
; CHECK: LV: Vector loop of width 2 costs: 35.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 2 for VF 4 For instruction:  %11 = zext i8
; CHECK: LV: Found an estimated cost of 24 for VF 4 For instruction: store i32
; CHECK: LV: Vector loop of width 4 costs: 20.
; CHECK: LV: Selecting VF: 4.
define hidden void @four_bytes_into_four_ints_vary_op(ptr noalias nocapture noundef writeonly %0, ptr nocapture noundef readonly %1, ptr nocapture noundef readonly %2, i32 noundef %3) {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %7, %4
  ret void

7:                                                ; preds = %4, %7
  %8 = phi i32 [ %40, %7 ], [ 0, %4 ]
  %9 = getelementptr inbounds %struct.FourBytes, ptr %1, i32 %8
  %10 = load i8, ptr %9, align 1
  %11 = zext i8 %10 to i32
  %12 = getelementptr inbounds %struct.FourBytes, ptr %2, i32 %8
  %13 = load i8, ptr %12, align 1
  %14 = zext i8 %13 to i32
  %15 = add nuw nsw i32 %14, %11
  %16 = getelementptr inbounds %struct.FourInts, ptr %0, i32 %8
  store i32 %15, ptr %16, align 4
  %17 = getelementptr inbounds i8, ptr %9, i32 1
  %18 = load i8, ptr %17, align 1
  %19 = zext i8 %18 to i32
  %20 = getelementptr inbounds i8, ptr %12, i32 1
  %21 = load i8, ptr %20, align 1
  %22 = zext i8 %21 to i32
  %23 = sub nsw i32 %19, %22
  %24 = getelementptr inbounds i8, ptr %16, i32 4
  store i32 %23, ptr %24, align 4
  %25 = getelementptr inbounds i8, ptr %9, i32 2
  %26 = load i8, ptr %25, align 1
  %27 = zext i8 %26 to i32
  %28 = getelementptr inbounds i8, ptr %12, i32 2
  %29 = load i8, ptr %28, align 1
  %30 = zext i8 %29 to i32
  %31 = mul nuw nsw i32 %30, %27
  %32 = getelementptr inbounds i8, ptr %16, i32 8
  store i32 %31, ptr %32, align 4
  %33 = getelementptr inbounds i8, ptr %9, i32 3
  %34 = load i8, ptr %33, align 1
  %35 = getelementptr inbounds i8, ptr %12, i32 3
  %36 = load i8, ptr %35, align 1
  %37 = and i8 %36, %34
  %38 = zext i8 %37 to i32
  %39 = getelementptr inbounds i8, ptr %16, i32 12
  store i32 %38, ptr %39, align 4
  %40 = add nuw i32 %8, 1
  %41 = icmp eq i32 %40, %3
  br i1 %41, label %6, label %7
}

; CHECK-LABEL: scale_uv_row_down2
; CHECK: LV: Scalar loop costs: 10.
; CHECK: LV: Vector loop of width 2 costs: 13.
; CHECK: LV: Vector loop of width 4 costs: 8.
; CHECK: LV: Vector loop of width 8 costs: 4.
; CHECK: LV: Vector loop of width 16 costs: 5.
; CHECK: LV: Selecting VF: 8.
define hidden void @scale_uv_row_down2(ptr nocapture noundef readonly %0, i32 noundef %1, ptr nocapture noundef writeonly %2, i32 noundef %3) {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %19

6:                                                ; preds = %4, %6
  %7 = phi i32 [ %17, %6 ], [ 0, %4 ]
  %8 = phi ptr [ %15, %6 ], [ %0, %4 ]
  %9 = phi ptr [ %16, %6 ], [ %2, %4 ]
  %10 = getelementptr inbounds i8, ptr %8, i32 2
  %11 = load i8, ptr %10, align 1
  store i8 %11, ptr %9, align 1
  %12 = getelementptr inbounds i8, ptr %8, i32 3
  %13 = load i8, ptr %12, align 1
  %14 = getelementptr inbounds i8, ptr %9, i32 1
  store i8 %13, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %8, i32 4
  %16 = getelementptr inbounds i8, ptr %9, i32 2
  %17 = add nuw nsw i32 %7, 1
  %18 = icmp eq i32 %17, %3
  br i1 %18, label %19, label %6

19:                                               ; preds = %6, %4
  ret void
}

; CHECK-LABEL: scale_uv_row_down2_box
; CHECK: LV: Scalar loop costs: 26.
; CHECK: LV: Vector loop of width 2 costs: 39.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %14 = load i8
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %20 = load i8
; CHECK: LV: Found an estimated cost of 11 for VF 4 For instruction: store i8 %48
; CHECK: LV: Vector loop of width 4 costs: 18.
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %14 = load i8
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %20 = load i8
; CHECK: LV: Found an estimated cost of 7 for VF 8 For instruction: store i8 %48
; CHECK: LV: Vector loop of width 8 costs: 10.
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %14 = load i8
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %20 = load i8
; CHECK: LV: Found an estimated cost of 6 for VF 16 For instruction: store i8 %48
; CHECK: LV: Vector loop of width 16 costs: 20.
; CHECK: LV: Selecting VF: 8.
define hidden void @scale_uv_row_down2_box(ptr nocapture noundef readonly %0, i32 noundef %1, ptr nocapture noundef writeonly %2, i32 noundef %3) {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %54

6:                                                ; preds = %4
  %7 = add nsw i32 %1, 2
  %8 = add nsw i32 %1, 1
  %9 = add nsw i32 %1, 3
  br label %10

10:                                               ; preds = %6, %10
  %11 = phi i32 [ 0, %6 ], [ %52, %10 ]
  %12 = phi ptr [ %0, %6 ], [ %50, %10 ]
  %13 = phi ptr [ %2, %6 ], [ %51, %10 ]
  %14 = load i8, ptr %12, align 1
  %15 = zext i8 %14 to i16
  %16 = getelementptr inbounds i8, ptr %12, i32 2
  %17 = load i8, ptr %16, align 1
  %18 = zext i8 %17 to i16
  %19 = getelementptr inbounds i8, ptr %12, i32 %1
  %20 = load i8, ptr %19, align 1
  %21 = zext i8 %20 to i16
  %22 = getelementptr inbounds i8, ptr %12, i32 %7
  %23 = load i8, ptr %22, align 1
  %24 = zext i8 %23 to i16
  %25 = add nuw nsw i16 %15, 2
  %26 = add nuw nsw i16 %25, %18
  %27 = add nuw nsw i16 %26, %21
  %28 = add nuw nsw i16 %27, %24
  %29 = lshr i16 %28, 2
  %30 = trunc nuw i16 %29 to i8
  store i8 %30, ptr %13, align 1
  %31 = getelementptr inbounds i8, ptr %12, i32 1
  %32 = load i8, ptr %31, align 1
  %33 = zext i8 %32 to i16
  %34 = getelementptr inbounds i8, ptr %12, i32 3
  %35 = load i8, ptr %34, align 1
  %36 = zext i8 %35 to i16
  %37 = getelementptr inbounds i8, ptr %12, i32 %8
  %38 = load i8, ptr %37, align 1
  %39 = zext i8 %38 to i16
  %40 = getelementptr inbounds i8, ptr %12, i32 %9
  %41 = load i8, ptr %40, align 1
  %42 = zext i8 %41 to i16
  %43 = add nuw nsw i16 %33, 2
  %44 = add nuw nsw i16 %43, %36
  %45 = add nuw nsw i16 %44, %39
  %46 = add nuw nsw i16 %45, %42
  %47 = lshr i16 %46, 2
  %48 = trunc nuw i16 %47 to i8
  %49 = getelementptr inbounds i8, ptr %13, i32 1
  store i8 %48, ptr %49, align 1
  %50 = getelementptr inbounds i8, ptr %12, i32 4
  %51 = getelementptr inbounds i8, ptr %13, i32 2
  %52 = add nuw nsw i32 %11, 1
  %53 = icmp eq i32 %52, %3
  br i1 %53, label %54, label %10

54:                                               ; preds = %10, %4
  ret void
}

; CHECK-LABEL: scale_uv_row_down2_linear
; CHECK: LV: Scalar loop costs: 18.
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: %13 = load i8
; CHECK: LV: Found an estimated cost of 6 for VF 2 For instruction: store i8
; CHECK: LV: Vector loop of width 2 costs: 25.
; CHECK: LV: Found an estimated cost of 18 for VF 4 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 11 for VF 4 For instruction: store i8
; CHECK: LV: Vector loop of width 4 costs: 11.
; CHECK: LV: Found an estimated cost of 26 for VF 8 For instruction: %10 = load i8
; CHECK: LV: Found an estimated cost of 7 for VF 8 For instruction: store i8
; CHECK: LV: Vector loop of width 8 costs: 6.
; CHECK: LV: Found an estimated cost of 132 for VF 16 For instruction: %10 = load i8
; CHECK: LV: Vector loop of width 16 costs: 10.
; CHECK: LV: Selecting VF: 8.
define hidden void @scale_uv_row_down2_linear(ptr nocapture noundef readonly %0, i32 noundef %1, ptr nocapture noundef writeonly %2, i32 noundef %3) {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %34

6:                                                ; preds = %4, %6
  %7 = phi i32 [ %32, %6 ], [ 0, %4 ]
  %8 = phi ptr [ %30, %6 ], [ %0, %4 ]
  %9 = phi ptr [ %31, %6 ], [ %2, %4 ]
  %10 = load i8, ptr %8, align 1
  %11 = zext i8 %10 to i16
  %12 = getelementptr inbounds i8, ptr %8, i32 2
  %13 = load i8, ptr %12, align 1
  %14 = zext i8 %13 to i16
  %15 = add nuw nsw i16 %11, 1
  %16 = add nuw nsw i16 %15, %14
  %17 = lshr i16 %16, 1
  %18 = trunc nuw i16 %17 to i8
  store i8 %18, ptr %9, align 1
  %19 = getelementptr inbounds i8, ptr %8, i32 1
  %20 = load i8, ptr %19, align 1
  %21 = zext i8 %20 to i16
  %22 = getelementptr inbounds i8, ptr %8, i32 3
  %23 = load i8, ptr %22, align 1
  %24 = zext i8 %23 to i16
  %25 = add nuw nsw i16 %21, 1
  %26 = add nuw nsw i16 %25, %24
  %27 = lshr i16 %26, 1
  %28 = trunc nuw i16 %27 to i8
  %29 = getelementptr inbounds i8, ptr %9, i32 1
  store i8 %28, ptr %29, align 1
  %30 = getelementptr inbounds i8, ptr %8, i32 4
  %31 = getelementptr inbounds i8, ptr %9, i32 2
  %32 = add nuw nsw i32 %7, 1
  %33 = icmp eq i32 %32, %3
  br i1 %33, label %34, label %6

34:                                               ; preds = %6, %4
  ret void
}
