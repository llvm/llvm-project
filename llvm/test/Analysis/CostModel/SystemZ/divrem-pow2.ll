; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 \
; RUN:  | FileCheck %s -check-prefix=COST

; Check that all divide/remainder instructions are implemented by cheaper instructions.
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -o - | FileCheck %s
; CHECK-NOT: dsg
; CHECK-NOT: dl

; Scalar sdiv

define i64 @fun0(i64 %a) {
  %r = sdiv i64 %a, 2
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i64 %a, 2
}

define i64 @fun1(i64 %a) {
  %r = sdiv i64 %a, -4
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i64 %a, -4
}

define i32 @fun2(i32 %a) {
  %r = sdiv i32 %a, 8
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i32 %a, 8
}

define i32 @fun3(i32 %a) {
  %r = sdiv i32 %a, -16
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i32 %a, -16
}

define i16 @fun4(i16 %a) {
  %r = sdiv i16 %a, 32
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i16 %a, 32
}

define i16 @fun5(i16 %a) {
  %r = sdiv i16 %a, -64
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i16 %a, -64
}

define i8 @fun6(i8 %a) {
  %r = sdiv i8 %a, 64
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i8 %a, 64
}

define i8 @fun7(i8 %a) {
  %r = sdiv i8 %a, -128
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv i8 %a, -128
}

; Vector sdiv

define <2 x i64> @fun8(<2 x i64> %a) {
  %r = sdiv <2 x i64> %a, <i64 2, i64 2>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <2 x i64> %a, <i64 2, i64 2>
}

define <2 x i64> @fun9(<2 x i64> %a) {
  %r = sdiv <2 x i64> %a, <i64 -4, i64 -4>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <2 x i64> %a, <i64 -4, i64 -4>
}

define <4 x i32> @fun10(<4 x i32> %a) {
  %r = sdiv <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
}

define <4 x i32> @fun11(<4 x i32> %a) {
  %r = sdiv <4 x i32> %a, <i32 -16, i32 -16, i32 -16, i32 -16>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <4 x i32> %a, <i32 -16
}

define <2 x i32> @fun12(<2 x i32> %a) {
  %r = sdiv <2 x i32> %a, <i32 -16, i32 -16>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <2 x i32> %a, <i32 -16
}

define <8 x i16> @fun13(<8 x i16> %a) {
  %r = sdiv <8 x i16> %a, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <8 x i16> %a, <i16 32
}

define <8 x i16> @fun14(<8 x i16> %a) {
  %r = sdiv <8 x i16> %a, <i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <8 x i16> %a, <i16 -64
}

define <4 x i16> @fun15(<4 x i16> %a) {
  %r = sdiv <4 x i16> %a, <i16 32, i16 32, i16 32, i16 32>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <4 x i16> %a, <i16 32
}

define <16 x i8> @fun16(<16 x i8> %a) {
  %r = sdiv <16 x i8> %a, <i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <16 x i8> %a, <i8 64
}

define <16 x i8> @fun17(<16 x i8> %a) {
  %r = sdiv <16 x i8> %a, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <16 x i8> %a, <i8 -128
}

define <8 x i8> @fun18(<8 x i8> %a) {
  %r = sdiv <8 x i8> %a, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = sdiv <8 x i8> %a, <i8 -128
}

; Scalar udiv

define i64 @fun19(i64 %a) {
  %r = udiv i64 %a, 2
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i64 %a, 2
}

define i32 @fun20(i32 %a) {
  %r = udiv i32 %a, 8
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i32 %a, 8
}

define i16 @fun21(i16 %a) {
  %r = udiv i16 %a, 32
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i16 %a, 32
}

define i8 @fun22(i8 %a) {
  %r = udiv i8 %a, 128
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv i8 %a, -128
}

; Vector udiv

define <2 x i64> @fun23(<2 x i64> %a) {
  %r = udiv <2 x i64> %a, <i64 2, i64 2>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <2 x i64> %a, <i64 2
}

define <4 x i32> @fun24(<4 x i32> %a) {
  %r = udiv <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <4 x i32> %a, <i32 8
}

define <2 x i32> @fun25(<2 x i32> %a) {
  %r = udiv <2 x i32> %a, <i32 8, i32 8>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <2 x i32> %a, <i32 8
}

define <8 x i16> @fun26(<8 x i16> %a) {
  %r = udiv <8 x i16> %a, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <8 x i16> %a, <i16 32
}

define <4 x i16> @fun27(<4 x i16> %a) {
  %r = udiv <4 x i16> %a, <i16 32, i16 32, i16 32, i16 32>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <4 x i16> %a, <i16 32
}

define <16 x i8> @fun28(<16 x i8> %a) {
  %r = udiv <16 x i8> %a, <i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <16 x i8> %a, <i8 -128
}

define <8 x i8> @fun29(<8 x i8> %a) {
  %r = udiv <8 x i8> %a, <i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = udiv <8 x i8> %a, <i8 -128
}

; Scalar srem

define i64 @fun30(i64 %a) {
  %r = srem i64 %a, 2
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i64 %a, 2
}

define i64 @fun31(i64 %a) {
  %r = srem i64 %a, -4
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i64 %a, -4
}

define i32 @fun32(i32 %a) {
  %r = srem i32 %a, 8
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i32 %a, 8
}

define i32 @fun33(i32 %a) {
  %r = srem i32 %a, -16
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i32 %a, -16
}

define i16 @fun34(i16 %a) {
  %r = srem i16 %a, 32
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i16 %a, 32
}

define i16 @fun35(i16 %a) {
  %r = srem i16 %a, -64
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i16 %a, -64
}

define i8 @fun36(i8 %a) {
  %r = srem i8 %a, 64
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i8 %a, 64
}

define i8 @fun37(i8 %a) {
  %r = srem i8 %a, -128
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem i8 %a, -128
}

; Vector srem

define <2 x i64> @fun38(<2 x i64> %a) {
  %r = srem <2 x i64> %a, <i64 2, i64 2>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <2 x i64> %a, <i64 2, i64 2>
}

define <2 x i64> @fun39(<2 x i64> %a) {
  %r = srem <2 x i64> %a, <i64 -4, i64 -4>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <2 x i64> %a, <i64 -4, i64 -4>
}

define <4 x i32> @fun40(<4 x i32> %a) {
  %r = srem <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
}

define <4 x i32> @fun41(<4 x i32> %a) {
  %r = srem <4 x i32> %a, <i32 -16, i32 -16, i32 -16, i32 -16>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <4 x i32> %a, <i32 -16
}

define <2 x i32> @fun42(<2 x i32> %a) {
  %r = srem <2 x i32> %a, <i32 -16, i32 -16>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <2 x i32> %a, <i32 -16
}

define <8 x i16> @fun43(<8 x i16> %a) {
  %r = srem <8 x i16> %a, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <8 x i16> %a, <i16 32
}

define <8 x i16> @fun44(<8 x i16> %a) {
  %r = srem <8 x i16> %a, <i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64, i16 -64>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <8 x i16> %a, <i16 -64
}

define <4 x i16> @fun45(<4 x i16> %a) {
  %r = srem <4 x i16> %a, <i16 32, i16 32, i16 32, i16 32>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <4 x i16> %a, <i16 32
}

define <16 x i8> @fun46(<16 x i8> %a) {
  %r = srem <16 x i8> %a, <i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64, i8 64>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <16 x i8> %a, <i8 64
}

define <16 x i8> @fun47(<16 x i8> %a) {
  %r = srem <16 x i8> %a, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <16 x i8> %a, <i8 -128
}

define <8 x i8> @fun48(<8 x i8> %a) {
  %r = srem <8 x i8> %a, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 4 for instruction:   %r = srem <8 x i8> %a, <i8 -128
}

; Scalar urem

define i64 @fun49(i64 %a) {
  %r = urem i64 %a, 2
  ret i64 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem i64 %a, 2
}

define i32 @fun50(i32 %a) {
  %r = urem i32 %a, 8
  ret i32 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem i32 %a, 8
}

define i16 @fun51(i16 %a) {
  %r = urem i16 %a, 32
  ret i16 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem i16 %a, 32
}

define i8 @fun52(i8 %a) {
  %r = urem i8 %a, 128
  ret i8 %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem i8 %a, -128
}

; Vector urem

define <2 x i64> @fun53(<2 x i64> %a) {
  %r = urem <2 x i64> %a, <i64 2, i64 2>
  ret <2 x i64> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <2 x i64> %a, <i64 2
}

define <4 x i32> @fun54(<4 x i32> %a) {
  %r = urem <4 x i32> %a, <i32 8, i32 8, i32 8, i32 8>
  ret <4 x i32> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <4 x i32> %a, <i32 8
}

define <2 x i32> @fun55(<2 x i32> %a) {
  %r = urem <2 x i32> %a, <i32 8, i32 8>
  ret <2 x i32> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <2 x i32> %a, <i32 8
}

define <8 x i16> @fun56(<8 x i16> %a) {
  %r = urem <8 x i16> %a, <i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32, i16 32>
  ret <8 x i16> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <8 x i16> %a, <i16 32
}

define <4 x i16> @fun57(<4 x i16> %a) {
  %r = urem <4 x i16> %a, <i16 32, i16 32, i16 32, i16 32>
  ret <4 x i16> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <4 x i16> %a, <i16 32
}

define <16 x i8> @fun58(<16 x i8> %a) {
  %r = urem <16 x i8> %a, <i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128>
  ret <16 x i8> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <16 x i8> %a, <i8 -128
}

define <8 x i8> @fun59(<8 x i8> %a) {
  %r = urem <8 x i8> %a, <i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128, i8 128>
  ret <8 x i8> %r
; COST: Cost Model: Found an estimated cost of 1 for instruction:   %r = urem <8 x i8> %a, <i8 -128
}
