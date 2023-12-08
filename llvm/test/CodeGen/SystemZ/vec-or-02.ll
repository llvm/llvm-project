; Test vector (or (and X, Z), (and Y, (not Z))) patterns.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8.
define <16 x i8> @f1(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3) {
; CHECK-LABEL: f1:
; CHECK: vsel %v24, %v24, %v26, %v28
; CHECK: br %r14
  %not = xor <16 x i8> %val3, <i8 -1, i8 -1, i8 -1, i8 -1,
                               i8 -1, i8 -1, i8 -1, i8 -1,
                               i8 -1, i8 -1, i8 -1, i8 -1,
                               i8 -1, i8 -1, i8 -1, i8 -1>
  %and1 = and <16 x i8> %val1, %val3
  %and2 = and <16 x i8> %val2, %not
  %ret = or <16 x i8> %and1, %and2
  ret <16 x i8> %ret
}

; ...and again with the XOR applied to the other operand of the AND.
define <16 x i8> @f2(<16 x i8> %val1, <16 x i8> %val2, <16 x i8> %val3) {
; CHECK-LABEL: f2:
; CHECK: vsel %v24, %v26, %v24, %v28
; CHECK: br %r14
  %not = xor <16 x i8> %val3, <i8 -1, i8 -1, i8 -1, i8 -1,
                               i8 -1, i8 -1, i8 -1, i8 -1,
                               i8 -1, i8 -1, i8 -1, i8 -1,
                               i8 -1, i8 -1, i8 -1, i8 -1>
  %and1 = and <16 x i8> %val1, %not
  %and2 = and <16 x i8> %val2, %val3
  %ret = or <16 x i8> %and1, %and2
  ret <16 x i8> %ret
}

; Test v8i16.
define <8 x i16> @f3(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3) {
; CHECK-LABEL: f3:
; CHECK: vsel %v24, %v24, %v26, %v28
; CHECK: br %r14
  %not = xor <8 x i16> %val3, <i16 -1, i16 -1, i16 -1, i16 -1,
                               i16 -1, i16 -1, i16 -1, i16 -1>
  %and1 = and <8 x i16> %val1, %val3
  %and2 = and <8 x i16> %val2, %not
  %ret = or <8 x i16> %and1, %and2
  ret <8 x i16> %ret
}

; ...and again with the XOR applied to the other operand of the AND.
define <8 x i16> @f4(<8 x i16> %val1, <8 x i16> %val2, <8 x i16> %val3) {
; CHECK-LABEL: f4:
; CHECK: vsel %v24, %v26, %v24, %v28
; CHECK: br %r14
  %not = xor <8 x i16> %val3, <i16 -1, i16 -1, i16 -1, i16 -1,
                               i16 -1, i16 -1, i16 -1, i16 -1>
  %and1 = and <8 x i16> %val1, %not
  %and2 = and <8 x i16> %val2, %val3
  %ret = or <8 x i16> %and1, %and2
  ret <8 x i16> %ret
}

; Test v4i32.
define <4 x i32> @f5(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3) {
; CHECK-LABEL: f5:
; CHECK: vsel %v24, %v24, %v26, %v28
; CHECK: br %r14
  %not = xor <4 x i32> %val3, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and1 = and <4 x i32> %val1, %val3
  %and2 = and <4 x i32> %val2, %not
  %ret = or <4 x i32> %and1, %and2
  ret <4 x i32> %ret
}

; ...and again with the XOR applied to the other operand of the AND.
define <4 x i32> @f6(<4 x i32> %val1, <4 x i32> %val2, <4 x i32> %val3) {
; CHECK-LABEL: f6:
; CHECK: vsel %v24, %v26, %v24, %v28
; CHECK: br %r14
  %not = xor <4 x i32> %val3, <i32 -1, i32 -1, i32 -1, i32 -1>
  %and1 = and <4 x i32> %val1, %not
  %and2 = and <4 x i32> %val2, %val3
  %ret = or <4 x i32> %and1, %and2
  ret <4 x i32> %ret
}

; Test v2i64.
define <2 x i64> @f7(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3) {
; CHECK-LABEL: f7:
; CHECK: vsel %v24, %v24, %v26, %v28
; CHECK: br %r14
  %not = xor <2 x i64> %val3, <i64 -1, i64 -1>
  %and1 = and <2 x i64> %val1, %val3
  %and2 = and <2 x i64> %val2, %not
  %ret = or <2 x i64> %and1, %and2
  ret <2 x i64> %ret
}

; ...and again with the XOR applied to the other operand of the AND.
define <2 x i64> @f8(<2 x i64> %val1, <2 x i64> %val2, <2 x i64> %val3) {
; CHECK-LABEL: f8:
; CHECK: vsel %v24, %v26, %v24, %v28
; CHECK: br %r14
  %not = xor <2 x i64> %val3, <i64 -1, i64 -1>
  %and1 = and <2 x i64> %val1, %not
  %and2 = and <2 x i64> %val2, %val3
  %ret = or <2 x i64> %and1, %and2
  ret <2 x i64> %ret
}
