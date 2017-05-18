; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

; Scalar sdiv

define i64 @fun0(i64 %a) {
  %r = sdiv i64 %a, 2
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i64 %a, 2
}

define i64 @fun1(i64 %a) {
  %r = sdiv i64 %a, -4
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i64 %a, -4
}

define i32 @fun2(i32 %a) {
  %r = sdiv i32 %a, 8
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i32 %a, 8
}

define i32 @fun3(i32 %a) {
  %r = sdiv i32 %a, -16
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i32 %a, -16
}

define i16 @fun4(i16 %a) {
  %r = sdiv i16 %a, 32
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i16 %a, 32
}

define i16 @fun5(i16 %a) {
  %r = sdiv i16 %a, -64
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i16 %a, -64
}

define i8 @fun6(i8 %a) {
  %r = sdiv i8 %a, 64
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i8 %a, 64
}

define i8 @fun7(i8 %a) {
  %r = sdiv i8 %a, -128
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i8 %a, -128
}


; Vector sdiv

define <2 x i64> @fun8(<2 x i64> %a) {
  %r = sdiv <2 x i64> %a, <i64 2, i64 2>
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <2 x i64> %a, <i64 2, i64 2>
}

define <2 x i64> @fun9(<2 x i64> %a) {
  %r = sdiv <2 x i64> %a, <i64 -4, i64 -4>
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <2 x i64> %a, <i64 -4, i64 -4>
}

define <4 x i32> @fun10(<4 x i32> %a) {
  %r = sdiv <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
}

define <4 x i32> @fun11(<4 x i32> %a) {
  %r = sdiv <4 x i32> %a, <i32 -16, i32 -16, i32 -16, i32 -16>
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <4 x i32> %a, <i32 -16
}

define <8 x i16> @fun12(<8 x i16> %a) {
  %r = sdiv <8 x i16> %a, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <8 x i16> %a, <i16 32
}

define <8 x i16> @fun13(<8 x i16> %a) {
  %r = sdiv <8 x i16> %a, <i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64>
  ret <8 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <8 x i16> %a, <i16 -64
}

define <16 x i8> @fun14(<16 x i8> %a) {
  %r = sdiv <16 x i8> %a, <i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64>
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <16 x i8> %a, <i8 64
}

define <16 x i8> @fun15(<16 x i8> %a) {
  %r = sdiv <16 x i8> %a, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <16 x i8> %a, <i8 -128
}

; Scalar udiv

define i64 @fun16(i64 %a) {
  %r = udiv i64 %a, 2
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i64 %a, 2
}

define i32 @fun17(i32 %a) {
  %r = udiv i32 %a, 8
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i32 %a, 8
}

define i16 @fun18(i16 %a) {
  %r = udiv i16 %a, 32
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i16 %a, 32
}

define i8 @fun19(i8 %a) {
  %r = udiv i8 %a, 128
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i8 %a, -128
}

; Vector udiv

define <2 x i64> @fun20(<2 x i64> %a) {
  %r = udiv <2 x i64> %a, <i64 2, i64 2>
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <2 x i64> %a, <i64 2
}

define <4 x i32> @fun21(<4 x i32> %a) {
  %r = udiv <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <4 x i32> %a, <i32 8
}

define <8 x i16> @fun22(<8 x i16> %a) {
  %r = udiv <8 x i16> %a, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <8 x i16> %a, <i16 32
}

define <16 x i8> @fun23(<16 x i8> %a) {
  %r = udiv <16 x i8> %a, <i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128>
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <16 x i8> %a, <i8 -128
}
