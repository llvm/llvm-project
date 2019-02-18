; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 \
; RUN:  | FileCheck %s -check-prefix=COST

; Check that all divide/remainder instructions are implemented by cheaper instructions.
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -o - | FileCheck %s
; CHECK-NOT: dsg
; CHECK-NOT: dl

; Check costs of divisions/remainders by a vector of constants that is *not*
; a power of two. A sequence containing a multiply and shifts will replace
; the divide instruction.

; Scalar sdiv

define i64 @fun0(i64 %a) {
  %r = sdiv i64 %a, 20
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = sdiv i64 %a, 20
}

define i32 @fun1(i32 %a) {
  %r = sdiv i32 %a, 20
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = sdiv i32 %a, 20
}

define i16 @fun2(i16 %a) {
  %r = sdiv i16 %a, 20
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = sdiv i16 %a, 20
}

define i8 @fun3(i8 %a) {
  %r = sdiv i8 %a, 20
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = sdiv i8 %a, 20
}

; Vector sdiv

define <2 x i64> @fun4(<2 x i64> %a) {
  %r = sdiv <2 x i64> %a, <i64 20, i64 21>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 24 for instruction:   %r = sdiv <2 x i64>
}

define <4 x i32> @fun5(<4 x i32> %a) {
  %r = sdiv <4 x i32> %a, <i32 20, i32 20, i32 20, i32 20>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = sdiv <4 x i32>
}

define <2 x i32> @fun6(<2 x i32> %a) {
  %r = sdiv <2 x i32> %a, <i32 20, i32 21>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 25 for instruction:   %r = sdiv <2 x i32>
}

define <8 x i16> @fun7(<8 x i16> %a) {
  %r = sdiv <8 x i16> %a, <i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = sdiv <8 x i16>
}

define <4 x i16> @fun8(<4 x i16> %a) {
  %r = sdiv <4 x i16> %a, <i16 20, i16 20, i16 20, i16 21>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = sdiv <4 x i16>
}

define <16 x i8> @fun9(<16 x i8> %a) {
  %r = sdiv <16 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 193 for instruction:   %r = sdiv <16 x i8>
}

define <8 x i8> @fun10(<8 x i8> %a) {
  %r = sdiv <8 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 21>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = sdiv <8 x i8>
}

; Scalar udiv

define i64 @fun11(i64 %a) {
  %r = udiv i64 %a, 20
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = udiv i64 %a, 20
}

define i32 @fun12(i32 %a) {
  %r = udiv i32 %a, 20
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = udiv i32 %a, 20
}

define i16 @fun13(i16 %a) {
  %r = udiv i16 %a, 20
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = udiv i16 %a, 20
}

define i8 @fun14(i8 %a) {
  %r = udiv i8 %a, 20
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = udiv i8
}

; Vector udiv

define <2 x i64> @fun15(<2 x i64> %a) {
  %r = udiv <2 x i64> %a, <i64 20, i64 20>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 24 for instruction:   %r = udiv <2 x i64>
}

define <4 x i32> @fun16(<4 x i32> %a) {
  %r = udiv <4 x i32> %a, <i32 20, i32 20, i32 20, i32 21>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = udiv <4 x i32>
}

define <2 x i32> @fun17(<2 x i32> %a) {
  %r = udiv <2 x i32> %a, <i32 20, i32 20>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 25 for instruction:   %r = udiv <2 x i32>
}

define <8 x i16> @fun18(<8 x i16> %a) {
  %r = udiv <8 x i16> %a, <i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 21>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = udiv <8 x i16>
}

define <4 x i16> @fun19(<4 x i16> %a) {
  %r = udiv <4 x i16> %a, <i16 20, i16 20, i16 20, i16 20>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = udiv <4 x i16>
}

define <16 x i8> @fun20(<16 x i8> %a) {
  %r = udiv <16 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 21>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 193 for instruction:   %r = udiv <16 x i8>
}

define <8 x i8> @fun21(<8 x i8> %a) {
  %r = udiv <8 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = udiv <8 x i8>
}

; Scalar srem

define i64 @fun22(i64 %a) {
  %r = srem i64 %a, 20
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = srem i64
}

define i32 @fun23(i32 %a) {
  %r = srem i32 %a, 20
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = srem i32
}

define i16 @fun24(i16 %a) {
  %r = srem i16 %a, 20
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = srem i16
}

define i8 @fun25(i8 %a) {
  %r = srem i8 %a, 20
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = srem i8
}

; Vector srem

define <2 x i64> @fun26(<2 x i64> %a) {
  %r = srem <2 x i64> %a, <i64 20, i64 21>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 24 for instruction:   %r = srem <2 x i64>
}

define <4 x i32> @fun27(<4 x i32> %a) {
  %r = srem <4 x i32> %a, <i32 20, i32 20, i32 20, i32 20>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = srem <4 x i32>
}

define <2 x i32> @fun28(<2 x i32> %a) {
  %r = srem <2 x i32> %a, <i32 20, i32 21>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 25 for instruction:   %r = srem <2 x i32>
}

define <8 x i16> @fun29(<8 x i16> %a) {
  %r = srem <8 x i16> %a, <i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = srem <8 x i16>
}

define <4 x i16> @fun30(<4 x i16> %a) {
  %r = srem <4 x i16> %a, <i16 20, i16 20, i16 20, i16 21>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = srem <4 x i16>
}

define <16 x i8> @fun31(<16 x i8> %a) {
  %r = srem <16 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 193 for instruction:   %r = srem <16 x i8>
}

define <8 x i8> @fun32(<8 x i8> %a) {
  %r = srem <8 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 21>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = srem <8 x i8>
}

; Scalar urem

define i64 @fun33(i64 %a) {
  %r = urem i64 %a, 20
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = urem i64
}

define i32 @fun34(i32 %a) {
  %r = urem i32 %a, 20
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = urem i32
}

define i16 @fun35(i16 %a) {
  %r = urem i16 %a, 20
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = urem i16
}

define i8 @fun36(i8 %a) {
  %r = urem i8 %a, 20
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 10 for instruction:   %r = urem i8
}

; Vector urem

define <2 x i64> @fun37(<2 x i64> %a) {
  %r = urem <2 x i64> %a, <i64 20, i64 20>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 24 for instruction:   %r = urem <2 x i64>
}

define <4 x i32> @fun38(<4 x i32> %a) {
  %r = urem <4 x i32> %a, <i32 20, i32 20, i32 20, i32 21>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = urem <4 x i32>
}

define <2 x i32> @fun39(<2 x i32> %a) {
  %r = urem <2 x i32> %a, <i32 20, i32 20>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 25 for instruction:   %r = urem <2 x i32>
}

define <8 x i16> @fun40(<8 x i16> %a) {
  %r = urem <8 x i16> %a, <i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 20, i16 21>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = urem <8 x i16>
}

define <4 x i16> @fun41(<4 x i16> %a) {
  %r = urem <4 x i16> %a, <i16 20, i16 20, i16 20, i16 20>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 49 for instruction:   %r = urem <4 x i16>
}

define <16 x i8> @fun42(<16 x i8> %a) {
  %r = urem <16 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 21>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 193 for instruction:   %r = urem <16 x i8>
}

define <8 x i8> @fun43(<8 x i8> %a) {
  %r = urem <8 x i8> %a, <i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20, i8 20>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 97 for instruction:   %r = urem <8 x i8>
}
