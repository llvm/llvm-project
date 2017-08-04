; Test vector NAND on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test a v16i8 NAND.
define <16 x i8> @f1(<16 x i8> %dummy, <16 x i8> %val1, <16 x i8> %val2) {
; CHECK-LABEL: f1:
; CHECK: vnn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <16 x i8> %val1, %val2
  %not = xor <16 x i8> %ret, <i8 -1, i8 -1, i8 -1, i8 -1,
                              i8 -1, i8 -1, i8 -1, i8 -1,
                              i8 -1, i8 -1, i8 -1, i8 -1,
                              i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %not
}

; Test a v8i16 NAND.
define <8 x i16> @f2(<8 x i16> %dummy, <8 x i16> %val1, <8 x i16> %val2) {
; CHECK-LABEL: f2:
; CHECK: vnn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <8 x i16> %val1, %val2
  %not = xor <8 x i16> %ret, <i16 -1, i16 -1, i16 -1, i16 -1,
                              i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %not
}

; Test a v4i32 NAND.
define <4 x i32> @f3(<4 x i32> %dummy, <4 x i32> %val1, <4 x i32> %val2) {
; CHECK-LABEL: f3:
; CHECK: vnn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <4 x i32> %val1, %val2
  %not = xor <4 x i32> %ret, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %not
}

; Test a v2i64 NAND.
define <2 x i64> @f4(<2 x i64> %dummy, <2 x i64> %val1, <2 x i64> %val2) {
; CHECK-LABEL: f4:
; CHECK: vnn %v24, %v26, %v28
; CHECK: br %r14
  %ret = and <2 x i64> %val1, %val2
  %not = xor <2 x i64> %ret, <i64 -1, i64 -1>
  ret <2 x i64> %not
}
