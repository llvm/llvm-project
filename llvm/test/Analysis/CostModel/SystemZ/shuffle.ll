; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

; CHECK: shuffle
define void @shuffle() {

  ;; Reverse shuffles
  shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  shufflevector <2 x i8> undef, <2 x i8> undef, <2 x i32> <i32 1, i32 0>

  shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  shufflevector <2 x i16> undef, <2 x i16> undef, <2 x i32> <i32 1, i32 0>

  shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 1, i32 0>

  shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 1, i32 0>

  shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
  shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 1, i32 0>

  ;; Alternate shuffles
  shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23, i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
  shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 16, i32 1, i32 18, i32 3, i32 20, i32 5, i32 22, i32 7, i32 24, i32 9, i32 26, i32 11, i32 28, i32 13, i32 30, i32 15>

  shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
  shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>

  shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
  shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 4, i32 1, i32 6, i32 3>

  shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 0, i32 3>
  shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 2, i32 1>

  shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 0, i32 3>
  shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 2, i32 1>

  ;; Broadcast shuffles
  shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> zeroinitializer
  shufflevector <32 x i8> undef, <32 x i8> undef, <32 x i32> zeroinitializer

  shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> zeroinitializer
  shufflevector <16 x i16> undef, <16 x i16> undef, <16 x i32> zeroinitializer

  shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> zeroinitializer
  shufflevector <8 x i32> undef, <8 x i32> undef, <8 x i32> zeroinitializer

  shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> zeroinitializer
  shufflevector <4 x i64> undef, <4 x i64> undef, <4 x i32> zeroinitializer

  shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> zeroinitializer
  shufflevector <4 x double> undef, <4 x double> undef, <4 x i32> zeroinitializer

  ;; Random shuffles
  shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 4, i32 17, i32 2, i32 19, i32 0, i32 21, i32 8, i32 23, i32 6, i32 10, i32 10, i32 27, i32 29, i32 29, i32 14, i32 31>
  shufflevector <18 x i8> undef, <18 x i8> undef, <18 x i32> <i32 4, i32 17, i32 2, i32 19, i32 0, i32 21, i32 8, i32 23, i32 6, i32 10, i32 10, i32 27, i32 29, i32 29, i32 14, i32 31, i32 0, i32 1>

  shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 9, i32 9, i32 2, i32 2, i32 4, i32 13, i32 15, i32 15>
  shufflevector <12 x i16> undef, <12 x i16> undef, <12 x i32> <i32 9, i32 9, i32 2, i32 2, i32 4, i32 13, i32 15, i32 15, i32 9, i32 2, i32 2, i32 4>

  shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 0, i32 0, i32 4, i32 7>
  shufflevector <6 x i32> undef, <6 x i32> undef, <6 x i32> <i32 0, i32 0, i32 4, i32 7, i32 4, i32 7>

  shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 1, i32 2>
  shufflevector <4 x i64> undef, <4 x i64> undef, <4 x i32> <i32 1, i32 2, i32 0, i32 2>

  shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 2, i32 1>
  shufflevector <4 x double> undef, <4 x double> undef, <4 x i32> <i32 2, i32 1, i32 0, i32 2>

  ret void

; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = shufflevector <2 x i8> undef, <2 x i8> undef, <2 x i32> <i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = shufflevector <2 x i16> undef, <2 x i16> undef, <2 x i32> <i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 1, i32 0>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %11 = shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 0, i32 17, i32 2, i32 19, i32 4, i32 21, i32 6, i32 23, i32 8, i32 25, i32 10, i32 27, i32 12, i32 29, i32 14, i32 31>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %12 = shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 16, i32 1, i32 18, i32 3, i32 20, i32 5, i32 22, i32 7, i32 24, i32 9, i32 26, i32 11, i32 28, i32 13, i32 30, i32 15>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %13 = shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %14 = shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 5, i32 14, i32 7>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %15 = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %16 = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %17 = shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 0, i32 3>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %18 = shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 2, i32 1>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %19 = shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 0, i32 3>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %20 = shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 2, i32 1>
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %21 = shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %22 = shufflevector <32 x i8> undef, <32 x i8> undef, <32 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %23 = shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %24 = shufflevector <16 x i16> undef, <16 x i16> undef, <16 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %25 = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %26 = shufflevector <8 x i32> undef, <8 x i32> undef, <8 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %27 = shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %28 = shufflevector <4 x i64> undef, <4 x i64> undef, <4 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %29 = shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %30 = shufflevector <4 x double> undef, <4 x double> undef, <4 x i32> zeroinitializer
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %31 = shufflevector <16 x i8> undef, <16 x i8> undef, <16 x i32> <i32 4, i32 17, i32 2, i32 19, i32 0, i32 21, i32 8, i32 23, i32 6, i32 10, i32 10, i32 27, i32 29, i32 29, i32 14, i32 31>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %32 = shufflevector <18 x i8> undef, <18 x i8> undef, <18 x i32> <i32 4, i32 17, i32 2, i32 19, i32 0, i32 21, i32 8, i32 23, i32 6, i32 10, i32 10, i32 27, i32 29, i32 29, i32 14, i32 31, i32 0, i32 1>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %33 = shufflevector <8 x i16> undef, <8 x i16> undef, <8 x i32> <i32 9, i32 9, i32 2, i32 2, i32 4, i32 13, i32 15, i32 15>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %34 = shufflevector <12 x i16> undef, <12 x i16> undef, <12 x i32> <i32 9, i32 9, i32 2, i32 2, i32 4, i32 13, i32 15, i32 15, i32 9, i32 2, i32 2, i32 4>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %35 = shufflevector <4 x i32> undef, <4 x i32> undef, <4 x i32> <i32 0, i32 0, i32 4, i32 7>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %36 = shufflevector <6 x i32> undef, <6 x i32> undef, <6 x i32> <i32 0, i32 0, i32 4, i32 7, i32 4, i32 7>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %37 = shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> <i32 1, i32 2>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %38 = shufflevector <4 x i64> undef, <4 x i64> undef, <4 x i32> <i32 1, i32 2, i32 0, i32 2>
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %39 = shufflevector <2 x double> undef, <2 x double> undef, <2 x i32> <i32 2, i32 1>
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %40 = shufflevector <4 x double> undef, <4 x double> undef, <4 x i32> <i32 2, i32 1, i32 0, i32 2>
}
