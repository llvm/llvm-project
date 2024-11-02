; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple=aarch64-linux-gnu -mattr=+neon < %s | FileCheck %s

; Verify the cost of integer division by constant.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

define <16 x i8> @sdiv8xi16(<16 x i8> %x) {
; CHECK-LABEL: function 'sdiv8xi16'
; CHECK: Found an estimated cost of 7 for instruction: %div = sdiv <16 x i8> %x, <i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9>
  %div = sdiv <16 x i8> %x, <i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9>
  ret <16 x i8> %div
}

define <8 x i16> @sdiv16xi8(<8 x i16> %x) {
; CHECK-LABEL: function 'sdiv16xi8'
; CHECK: Found an estimated cost of 7 for instruction: %div = sdiv <8 x i16> %x, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
  %div = sdiv <8 x i16> %x, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
  ret <8 x i16> %div
}

define <4 x i32> @sdiv32xi4(<4 x i32> %x) {
; CHECK-LABEL: function 'sdiv32xi4'
; CHECK: Found an estimated cost of 7 for instruction: %div = sdiv <4 x i32> %x, <i32 9, i32 9, i32 9, i32 9>
  %div = sdiv <4 x i32> %x, <i32 9, i32 9, i32 9, i32 9>
  ret <4 x i32> %div
}

define <16 x i8> @udiv8xi16(<16 x i8> %x) {
; CHECK-LABEL: function 'udiv8xi16'
; CHECK: Found an estimated cost of 7 for instruction: %div = udiv <16 x i8> %x, <i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9>
  %div = udiv <16 x i8> %x, <i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9, i8 9>
  ret <16 x i8> %div
}

define <8 x i16> @udiv16xi8(<8 x i16> %x) {
; CHECK-LABEL: function 'udiv16xi8'
; CHECK: Found an estimated cost of 7 for instruction:   %div = udiv <8 x i16> %x, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
  %div = udiv <8 x i16> %x, <i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9, i16 9>
  ret <8 x i16> %div
}

define <4 x i32> @udiv32xi4(<4 x i32> %x) {
; CHECK-LABEL: function 'udiv32xi4'
; CHECK: Found an estimated cost of 7 for instruction:   %div = udiv <4 x i32> %x, <i32 9, i32 9, i32 9, i32 9>
  %div = udiv <4 x i32> %x, <i32 9, i32 9, i32 9, i32 9>
  ret <4 x i32> %div
}
