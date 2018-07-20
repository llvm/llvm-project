; RUN: opt -cost-model -analyze -mtriple=aarch64--linux-gnu < %s | FileCheck %s

; Verify the cost of integer division instructions.

define i32 @sdivs1i32(i32 %a, i32 %b) {
; CHECK-LABEL: 'Cost Model Analysis' for function 'sdivs1i32':
; CHECK: Found an estimated cost of 1 for instruction: %c = sdiv i32 %a, %b
  %c = sdiv i32 %a, %b
  ret i32 %c
}

define i64 @sdivs1i64(i64 %a, i64 %b) {
; CHECK-LABEL: 'Cost Model Analysis' for function 'sdivs1i64':
; CHECK: Found an estimated cost of 1 for instruction: %c = sdiv i64 %a, %b
  %c = sdiv i64 %a, %b
  ret i64 %c
}

define <2 x i32> @sdivv2i32(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: 'Cost Model Analysis' for function 'sdivv2i32':
; CHECK: Found an estimated cost of 24 for instruction: %c = sdiv <2 x i32> %a, %b
  %c = sdiv <2 x i32> %a, %b
  ret <2 x i32> %c
}

define <2 x i64> @sdivv2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK-LABEL: 'Cost Model Analysis' for function 'sdivv2i64':
; CHECK: Found an estimated cost of 24 for instruction: %c = sdiv <2 x i64> %a, %b
  %c = sdiv <2 x i64> %a, %b
  ret <2 x i64> %c
}

define <4 x i32> @sdivv4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK-LABEL: 'Cost Model Analysis' for function 'sdivv4i32':
; CHECK: Found an estimated cost of 52 for instruction: %c = sdiv <4 x i32> %a, %b
  %c = sdiv <4 x i32> %a, %b
  ret <4 x i32> %c
}
