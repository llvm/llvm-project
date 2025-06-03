; RUN: llc -mtriple=hexagon -O2 -disable-hexagon-amodeopt < %s | FileCheck %s --check-prefix=CHECK-ADDI
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { i8, i8, %s.41, %s.1, %s.2, i8, %s.22, i8, %s.3, i8, i8, %s.23, %s.23, %s.4, i8, %s.5, %s.6, %s.10, %s.14, %s.44, i16, i8, i32, i16, i16, %s.16, i8, i8, i16, i8, i8, i32, i8, [8 x %s.17], i8, i8, i8, i8, i8, i64, i64, i64, i8, i8, i8, i8, i8, i8, i16, i16, i8, i8, i16, i16, i16, i16, i16, i8, i8, i32, i8, i32, i32, i8, [256 x %s.22], [256 x i8], i8, i8, %s.18, i8, i8, i8, i8, i16, i16, i16, i8, i32, i8, i8, i8, i8, i16, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i16, i8, i8, i8, i8, i8, i8, i32, i64, i64, i8, %s.22, %s.23, i8, i8, i8, i8, i8, i8, i8, i16, i32, [256 x %s.22], i8, i8, %s.25, %s.26, i8, i8, %s.27, i8, i8, i8, i8, i8, i8, i8, i8, %s.41, i8, i8, i8, %s.28, i8, %s.30, %s.33, %s.33, %s.33, %s.33, %s.33, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, %s.34, i8, i8, %s.38, i8, i8, %s.40, i8, i8, i8, i8, i8, %s.41, i8, i8, i8, i32, %s.42, i8, i16, [32 x i16], i8, i8, %s.43, i8, i8, i8, i8, i8, i8, i8, i8, %s.44, i8, i8, i8, i8, i32, i8, i8, i8, i8, i8 }
%s.1 = type { i32, [30 x %s.16] }
%s.2 = type { [10 x %s.27], i8, i8 }
%s.3 = type { i8, %s.41 }
%s.4 = type { i8, i8, ptr }
%s.5 = type { %s.22, %s.1, i8, i8, i64, i8, i8, i32, i8, i8, i8, %s.34, i8, i8, i32, i8 }
%s.6 = type { i64, i8, i8, %s.7, i8, i8, %s.34, %s.34, i8, i8, %s.26 }
%s.7 = type { i32, [256 x %s.8] }
%s.8 = type { %s.9, i8 }
%s.9 = type { [3 x i8] }
%s.10 = type { i32, [40 x %s.11] }
%s.11 = type { %s.41, i8, i8, i8, i32, %s.12, i32 }
%s.12 = type { i32, %s.13, i8 }
%s.13 = type { i8, [48 x i8] }
%s.14 = type { i8, [10 x %s.15] }
%s.15 = type { i16, i8, %s.41, i8, i8 }
%s.16 = type { %s.41, [2 x i8] }
%s.17 = type { i8, i32 }
%s.18 = type { %s.19, i8, %s.20, i8, i8 }
%s.19 = type { i8, i8, i8, i8 }
%s.20 = type { i32, [40 x %s.21] }
%s.21 = type { %s.9, i8, i8, i8, i8, i8, i32, %s.12, i32 }
%s.22 = type { i8, %s.41, i32 }
%s.23 = type { %s.41, i16, i16, i16, i32, i8, i16, i8, i16, i8, [8 x i8], i8, i8, i8, %s.24, i8, %s.28 }
%s.24 = type { %s.16, [1 x i8] }
%s.25 = type { i64, i64, i64, i64 }
%s.26 = type { i8, i8, i8, i8, [12 x i8] }
%s.27 = type { %s.9, [2 x i8] }
%s.28 = type { %s.41, [6 x %s.29], i8 }
%s.29 = type { %s.41, i16 }
%s.30 = type { i8, [16 x %s.31] }
%s.31 = type { i32, i16, i8, i8, [32 x %s.32] }
%s.32 = type { i32, i8, i8 }
%s.33 = type { i8, [16 x i16] }
%s.34 = type { i32, i32, [10 x %s.35], %s.37 }
%s.35 = type { %s.36, i8, i32, i8, %s.36 }
%s.36 = type { %s.25 }
%s.37 = type { i8, i8 }
%s.38 = type { i16, [64 x %s.39] }
%s.39 = type { i16, i8, i16 }
%s.40 = type { i8, [3 x i8], i8 }
%s.41 = type { [3 x i8], i8, [3 x i8] }
%s.42 = type { i16, i16, [32 x i16], [32 x %s.41], [32 x i8] }
%s.43 = type { i8, i8, i8, i8, [9 x i16] }
%s.44 = type { %s.45, %s.47 }
%s.45 = type { %s.46, i32, i8 }
%s.46 = type { ptr, ptr }
%s.47 = type { %s.48 }
%s.48 = type { %s.46, %s.49 }
%s.49 = type { %s.50 }
%s.50 = type { %s.51, [16 x %s.52], i8, i8, [16 x i16], %s.9, i8, %s.59, %s.33, %s.62, %s.34, %s.64, i8 }
%s.51 = type { i32, i16, i8, i8, i8, i8, i8, [5 x i8] }
%s.52 = type { i8, %s.53 }
%s.53 = type { %s.54 }
%s.54 = type { ptr, i8, i32 }
%s.55 = type { %s.46, i32, ptr, ptr, ptr, ptr, i32, i8, i8, i16, i32, i8, %s.56, i16, [1 x %s.58], i32 }
%s.56 = type { %s.57 }
%s.57 = type { i8 }
%s.58 = type { ptr, i32 }
%s.59 = type { i8, [17 x %s.60] }
%s.60 = type { i16, i8, [16 x %s.61] }
%s.61 = type { i8, i8 }
%s.62 = type { i8, [6 x %s.63] }
%s.63 = type { i8, i16 }
%s.64 = type { %s.65, i8, i64 }
%s.65 = type { i32, [64 x %s.66], i32, [64 x %s.66], i32, [64 x %s.66], i32, [128 x %s.66], i32, [32 x %s.67], i32, [32 x %s.67] }
%s.66 = type { i8, i32, i8 }
%s.67 = type { i16, i8 }
%s.68 = type { %s.69 }
%s.69 = type { i32, ptr }

@g0 = external global %s.0, align 8
@g1 = external constant %s.68, section ".dummy.dummy.dummy.dumm", align 4

; Function Attrs: optsize
declare void @f0(ptr) #0

; Function Attrs: nounwind optsize
declare zeroext i8 @f1(ptr) #1

; Function Attrs: nounwind optsize
declare void @f2(i32) #1

; The pass that used to crash doesn't do anything on this testcase anymore,
; but check for sane output anyway.
; CHECK-ADDI: ##g0
; Function Attrs: nounwind optsize ssp
define zeroext i8 @f3() #2 {
b0:
  %v0 = load i8, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 57), align 2
  %v1 = icmp eq i8 %v0, 0
  br i1 %v1, label %b2, label %b1

b1:                                               ; preds = %b0
  tail call void @f0(ptr nonnull @g1) #3
  unreachable

b2:                                               ; preds = %b0
  %v2 = call zeroext i8 @f1(ptr nonnull undef) #4
  br i1 undef, label %b3, label %b8

b3:                                               ; preds = %b2
  %v3 = load i8, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 1), align 1
  %v4 = add i8 %v3, -17
  %v5 = icmp ult i8 %v4, 2
  br i1 %v5, label %b4, label %b7

b4:                                               ; preds = %b3
  %v6 = load i8, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 167, i32 2), align 2
  %v7 = sext i8 %v6 to i32
  %v8 = add nsw i32 %v7, 1
  %v9 = load i8, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 167, i32 0), align 2
  %v10 = zext i8 %v9 to i32
  %v11 = icmp slt i32 %v8, %v10
  br i1 %v11, label %b6, label %b5

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b4
  unreachable

b7:                                               ; preds = %b3
  unreachable

b8:                                               ; preds = %b2
  br i1 undef, label %b9, label %b10

b9:                                               ; preds = %b8
  unreachable

b10:                                              ; preds = %b8
  br i1 undef, label %b12, label %b11

b11:                                              ; preds = %b10
  unreachable

b12:                                              ; preds = %b10
  %v12 = load i8, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 1), align 1
  %v13 = zext i8 %v12 to i32
  switch i32 %v13, label %b14 [
    i32 17, label %b13
    i32 18, label %b13
    i32 11, label %b15
  ]

b13:                                              ; preds = %b14, %b12, %b12
  %v14 = phi i64 [ 4294967294, %b14 ], [ 4294967146, %b12 ], [ 4294967146, %b12 ]
  %v15 = call i64 @f4(i8 zeroext undef) #3
  %v16 = add i64 %v15, %v14
  %v17 = trunc i64 %v16 to i32
  br label %b15

b14:                                              ; preds = %b12
  br label %b13

b15:                                              ; preds = %b13, %b12
  %v18 = phi i32 [ %v17, %b13 ], [ 120000, %b12 ]
  call void @f2(i32 %v18) #4
  unreachable
}

; Function Attrs: optsize
declare i64 @f4(i8 zeroext) #0

attributes #0 = { optsize "target-cpu"="hexagonv55" }
attributes #1 = { nounwind optsize "target-cpu"="hexagonv55" }
attributes #2 = { nounwind optsize ssp "target-cpu"="hexagonv55" }
attributes #3 = { nounwind optsize }
attributes #4 = { noinline nounwind optsize }
