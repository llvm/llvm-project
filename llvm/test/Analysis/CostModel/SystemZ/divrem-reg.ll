; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

; Check costs of divisions by register
;
; Note: Vectorization of division/remainder is temporarily disabled for high
; vectorization factors by returning 1000.

; Scalar sdiv

define i64 @fun0(i64 %a, i64 %b) {
  %r = sdiv i64 %a, %b
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = sdiv i64
}

define i32 @fun1(i32 %a, i32 %b) {
  %r = sdiv i32 %a, %b
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = sdiv i32 %a, %b
}

define i16 @fun2(i16 %a, i16 %b) {
  %r = sdiv i16 %a, %b
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = sdiv i16 %a, %b
}

define i8 @fun3(i8 %a, i8 %b) {
  %r = sdiv i8 %a, %b
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = sdiv i8 %a, %b
}

; Vector sdiv

define <2 x i64> @fun4(<2 x i64> %a, <2 x i64> %b) {
  %r = sdiv <2 x i64> %a, %b
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 47 for instruction:   %r = sdiv <2 x i64>
}

define <4 x i32> @fun5(<4 x i32> %a, <4 x i32> %b) {
  %r = sdiv <4 x i32> %a, %b
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = sdiv <4 x i32>
}

define <2 x i32> @fun6(<2 x i32> %a, <2 x i32> %b) {
  %r = sdiv <2 x i32> %a, %b
  ret <2 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %r = sdiv <2 x i32>
}

define <8 x i16> @fun7(<8 x i16> %a, <8 x i16> %b) {
  %r = sdiv <8 x i16> %a, %b
  ret <8 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = sdiv <8 x i16>
}

define <4 x i16> @fun8(<4 x i16> %a, <4 x i16> %b) {
  %r = sdiv <4 x i16> %a, %b
  ret <4 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = sdiv <4 x i16>
}

define <16 x i8> @fun9(<16 x i8> %a, <16 x i8> %b) {
  %r = sdiv <16 x i8> %a, %b
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = sdiv <16 x i8>
}

define <8 x i8> @fun10(<8 x i8> %a, <8 x i8> %b) {
  %r = sdiv <8 x i8> %a, %b
  ret <8 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = sdiv <8 x i8>
}

; Scalar udiv

define i64 @fun11(i64 %a, i64 %b) {
  %r = udiv i64 %a, %b
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = udiv i64 %a, %b
}

define i32 @fun12(i32 %a, i32 %b) {
  %r = udiv i32 %a, %b
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = udiv i32
}

define i16 @fun13(i16 %a, i16 %b) {
  %r = udiv i16 %a, %b
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = udiv i16
}

define i8 @fun14(i8 %a, i8 %b) {
  %r = udiv i8 %a, %b
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = udiv i8
}

; Vector udiv

define <2 x i64> @fun15(<2 x i64> %a, <2 x i64> %b) {
  %r = udiv <2 x i64> %a, %b
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 47 for instruction:   %r = udiv <2 x i64>
}

define <4 x i32> @fun16(<4 x i32> %a, <4 x i32> %b) {
  %r = udiv <4 x i32> %a, %b
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = udiv <4 x i32>
}

define <2 x i32> @fun17(<2 x i32> %a, <2 x i32> %b) {
  %r = udiv <2 x i32> %a, %b
  ret <2 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %r = udiv <2 x i32>
}

define <8 x i16> @fun18(<8 x i16> %a, <8 x i16> %b) {
  %r = udiv <8 x i16> %a, %b
  ret <8 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = udiv <8 x i16>
}

define <4 x i16> @fun19(<4 x i16> %a, <4 x i16> %b) {
  %r = udiv <4 x i16> %a, %b
  ret <4 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = udiv <4 x i16>
}

define <16 x i8> @fun20(<16 x i8> %a, <16 x i8> %b) {
  %r = udiv <16 x i8> %a, %b
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = udiv <16 x i8>
}

define <8 x i8> @fun21(<8 x i8> %a, <8 x i8> %b) {
  %r = udiv <8 x i8> %a, %b
  ret <8 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = udiv <8 x i8>
}

; Scalar srem

define i64 @fun22(i64 %a, i64 %b) {
  %r = srem i64 %a, %b
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = srem i64
}

define i32 @fun23(i32 %a, i32 %b) {
  %r = srem i32 %a, %b
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = srem i32
}

define i16 @fun24(i16 %a, i16 %b) {
  %r = srem i16 %a, %b
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = srem i16
}

define i8 @fun25(i8 %a, i8 %b) {
  %r = srem i8 %a, %b
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = srem i8
}

; Vector srem

define <2 x i64> @fun26(<2 x i64> %a, <2 x i64> %b) {
  %r = srem <2 x i64> %a, %b
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 47 for instruction:   %r = srem <2 x i64>
}

define <4 x i32> @fun27(<4 x i32> %a, <4 x i32> %b) {
  %r = srem <4 x i32> %a, %b
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = srem <4 x i32>
}

define <2 x i32> @fun28(<2 x i32> %a, <2 x i32> %b) {
  %r = srem <2 x i32> %a, %b
  ret <2 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %r = srem <2 x i32>
}

define <8 x i16> @fun29(<8 x i16> %a, <8 x i16> %b) {
  %r = srem <8 x i16> %a, %b
  ret <8 x i16> %r
; CHECK: ost Model: Found an estimated cost of 1000 for instruction:   %r = srem <8 x i16>
}

define <4 x i16> @fun30(<4 x i16> %a, <4 x i16> %b) {
  %r = srem <4 x i16> %a, %b
  ret <4 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = srem <4 x i16>
}

define <16 x i8> @fun31(<16 x i8> %a, <16 x i8> %b) {
  %r = srem <16 x i8> %a, %b
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = srem <16 x i8>
}

define <8 x i8> @fun32(<8 x i8> %a, <8 x i8> %b) {
  %r = srem <8 x i8> %a, %b
  ret <8 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = srem <8 x i8>
}

; Scalar urem

define i64 @fun33(i64 %a, i64 %b) {
  %r = urem i64 %a, %b
  ret i64 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = urem i64
}

define i32 @fun34(i32 %a, i32 %b) {
  %r = urem i32 %a, %b
  ret i32 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = urem i32
}

define i16 @fun35(i16 %a, i16 %b) {
  %r = urem i16 %a, %b
  ret i16 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = urem i16
}

define i8 @fun36(i8 %a, i8 %b) {
  %r = urem i8 %a, %b
  ret i8 %r
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %r = urem i8
}

; Vector urem

define <2 x i64> @fun37(<2 x i64> %a, <2 x i64> %b) {
  %r = urem <2 x i64> %a, %b
  ret <2 x i64> %r
; CHECK: Cost Model: Found an estimated cost of 47 for instruction:   %r = urem <2 x i64>
}

define <4 x i32> @fun38(<4 x i32> %a, <4 x i32> %b) {
  %r = urem <4 x i32> %a, %b
  ret <4 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = urem <4 x i32>
}

define <2 x i32> @fun39(<2 x i32> %a, <2 x i32> %b) {
  %r = urem <2 x i32> %a, %b
  ret <2 x i32> %r
; CHECK: Cost Model: Found an estimated cost of 48 for instruction:   %r = urem <2 x i32>
}

define <8 x i16> @fun40(<8 x i16> %a, <8 x i16> %b) {
  %r = urem <8 x i16> %a, %b
  ret <8 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = urem <8 x i16>
}

define <4 x i16> @fun41(<4 x i16> %a, <4 x i16> %b) {
  %r = urem <4 x i16> %a, %b
  ret <4 x i16> %r
; CHECK: Cost Model: Found an estimated cost of 94 for instruction:   %r = urem <4 x i16>
}

define <16 x i8> @fun42(<16 x i8> %a, <16 x i8> %b) {
  %r = urem <16 x i8> %a, %b
  ret <16 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = urem <16 x i8>
}

define <8 x i8> @fun43(<8 x i8> %a, <8 x i8> %b) {
  %r = urem <8 x i8> %a, %b
  ret <8 x i8> %r
; CHECK: Cost Model: Found an estimated cost of 1000 for instruction:   %r = urem <8 x i8>
}
