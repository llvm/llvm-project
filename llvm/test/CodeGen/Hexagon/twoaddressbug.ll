; RUN: llc -mtriple=hexagon -O2 < %s
; REQUIRES: asserts

; This file used to fail with an "UNREACHABLE executed!" in Post-RA pseudo
; instruction expansion pass due to a bug in the TwoAddressInstructionPass we
; were not handling sub register indexes when rewriting tied operands.

target triple = "hexagon"

%0 = type { i8, i8, %1, i32, i32, %7, i8, i8, %8, i8, i32, i16, i16, [2500 x i8], i16, i16, i16, ptr, [1024 x i8], i32, i32, i32, i32, i32, i8 }
%1 = type { i8, %2, i8, i8, i32 }
%2 = type { %3 }
%3 = type { i8, [256 x i8], %4, i8, i16, i32 }
%4 = type { %5 }
%5 = type { %6 }
%6 = type { [2 x i64] }
%7 = type { i32, i8 }
%8 = type { %7, i32, i32, %1 }
%9 = type { %10, ptr }
%10 = type { i16, i16, i32 }
%11 = type { i8, i32 }

@g0 = external hidden global [2 x %0], align 8
@g1 = external hidden constant %9, align 4
@g2 = external hidden constant %9, align 4
@g3 = external hidden constant %9, align 4
@g4 = external hidden constant %9, align 4

; Function Attrs: optsize
declare void @f0(ptr, i32, i32, i32) #0

; Function Attrs: nounwind optsize ssp
define hidden fastcc void @f1(i64 %a0, i8 zeroext %a1, i8 zeroext %a2) #1 {
b0:
  %v0 = alloca %11, align 4
  %v1 = icmp ne i8 %a1, 0
  %v2 = trunc i64 %a0 to i32
  br i1 %v1, label %b1, label %b4

b1:                                               ; preds = %b0
  call void @f0(ptr @g1, i32 %v2, i32 0, i32 0) #2
  %v3 = getelementptr inbounds [2 x %0], ptr @g0, i32 0, i32 %v2, i32 7
  store i8 1, ptr %v3, align 1
  %v4 = icmp eq i8 %a2, 0
  br i1 %v4, label %b4, label %b2

b2:                                               ; preds = %b1
  store i8 0, ptr %v0, align 4
  %v6 = getelementptr inbounds %11, ptr %v0, i32 0, i32 1
  store i32 0, ptr %v6, align 4
  %v7 = getelementptr inbounds [2 x %0], ptr @g0, i32 0, i32 %v2, i32 3
  %v8 = load i32, ptr %v7, align 8
  %v9 = getelementptr inbounds [2 x %0], ptr @g0, i32 0, i32 %v2, i32 4
  %v10 = load i32, ptr %v9, align 4
  %v11 = getelementptr inbounds [2 x %0], ptr @g0, i32 0, i32 %v2, i32 19
  %v12 = load i32, ptr %v11, align 4
  %v13 = call zeroext i8 @f2(i64 %a0, i32 %v8, i32 %v10, i32 %v12, i8 zeroext 0, ptr %v0) #2
  %v14 = icmp eq i8 %v13, 0
  br i1 %v14, label %b4, label %b3

b3:                                               ; preds = %b2
  %v15 = zext i8 %v13 to i32
  call void @f0(ptr @g2, i32 %v15, i32 %v2, i32 0) #2
  br label %b4

b4:                                               ; preds = %b3, %b2, %b1, %b0
  %v16 = getelementptr inbounds [2 x %0], ptr @g0, i32 0, i32 %v2, i32 1
  %v17 = load i8, ptr %v16, align 1
  %v18 = zext i8 %v17 to i32
  switch i32 %v18, label %b14 [
    i32 2, label %b11
    i32 1, label %b5
    i32 4, label %b8
    i32 3, label %b11
  ]

b5:                                               ; preds = %b4
  call void @f0(ptr @g3, i32 %v2, i32 0, i32 0) #2
  br i1 %v1, label %b7, label %b6

b6:                                               ; preds = %b5
  call fastcc void @f3(i64 %a0, i8 zeroext 0, i8 zeroext 1, i32 1) #0
  br label %b14

b7:                                               ; preds = %b5
  call fastcc void @f3(i64 %a0, i8 zeroext 0, i8 zeroext 0, i32 1) #0
  br label %b14

b8:                                               ; preds = %b4
  call void @f0(ptr @g4, i32 %v2, i32 0, i32 0) #2
  %v19 = getelementptr inbounds [2 x %0], ptr @g0, i32 0, i32 %v2, i32 6
  store i8 1, ptr %v19, align 8
  br i1 %v1, label %b10, label %b9

b9:                                               ; preds = %b8
  call fastcc void @f3(i64 %a0, i8 zeroext 0, i8 zeroext 1, i32 1) #0
  br label %b14

b10:                                              ; preds = %b8
  call fastcc void @f3(i64 %a0, i8 zeroext 0, i8 zeroext 0, i32 1) #0
  br label %b14

b11:                                              ; preds = %b4, %b4
  br i1 %v1, label %b13, label %b12

b12:                                              ; preds = %b11
  call fastcc void @f3(i64 %a0, i8 zeroext 0, i8 zeroext 1, i32 1) #0
  br label %b14

b13:                                              ; preds = %b11
  call fastcc void @f3(i64 %a0, i8 zeroext 0, i8 zeroext 0, i32 1) #0
  br label %b14

b14:                                              ; preds = %b13, %b12, %b10, %b9, %b7, %b6, %b4
  ret void
}

; Function Attrs: optsize
declare zeroext i8 @f2(i64, i32, i32, i32, i8 zeroext, ptr) #0

; Function Attrs: nounwind optsize ssp
declare hidden fastcc void @f3(i64, i8 zeroext, i8 zeroext, i32) #1

attributes #0 = { optsize }
attributes #1 = { nounwind optsize ssp }
attributes #2 = { nounwind optsize }
