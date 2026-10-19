; RUN: opt < %s -mtriple=aarch64--linux-gnu -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefix=COST
; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s --check-prefix=CODE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; COST-LABEL: uaddl_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <8 x i8> %b to <8 x i16>
; CODE-LABEL: uaddl_8h
; CODE:       uaddl v0.8h, v0.8b, v1.8b
define <8 x i16> @uaddl_8h(<8 x i8> %a, <8 x i8> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = zext <8 x i8> %b to <8 x i16>
  %tmp2 = add <8 x i16> %tmp0, %tmp1
  ret <8 x i16> %tmp2
}

; COST-LABEL: uaddl_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i16> %a to <4 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <4 x i16> %b to <4 x i32>
; CODE-LABEL: uaddl_4s
; CODE:       uaddl v0.4s, v0.4h, v1.4h
define <4 x i32> @uaddl_4s(<4 x i16> %a, <4 x i16> %b) {
  %tmp0 = zext <4 x i16> %a to <4 x i32>
  %tmp1 = zext <4 x i16> %b to <4 x i32>
  %tmp2 = add <4 x i32> %tmp0, %tmp1
  ret <4 x i32> %tmp2
}

; COST-LABEL: uaddl_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <2 x i32> %a to <2 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <2 x i32> %b to <2 x i64>
; CODE-LABEL: uaddl_2d
; CODE:       uaddl v0.2d, v0.2s, v1.2s
define <2 x i64> @uaddl_2d(<2 x i32> %a, <2 x i32> %b) {
  %tmp0 = zext <2 x i32> %a to <2 x i64>
  %tmp1 = zext <2 x i32> %b to <2 x i64>
  %tmp2 = add <2 x i64> %tmp0, %tmp1
  ret <2 x i64> %tmp2
}

; COST-LABEL: uaddl2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <16 x i8> %a to <16 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <16 x i8> %b to <16 x i16>
; CODE-LABEL: uaddl2_8h
; CODE:       uaddl2 v2.8h, v0.16b, v1.16b
; CODE-NEXT:  uaddl v0.8h, v0.8b, v1.8b
define <16 x i16> @uaddl2_8h(<16 x i8> %a, <16 x i8> %b) {
  %tmp0 = zext <16 x i8> %a to <16 x i16>
  %tmp1 = zext <16 x i8> %b to <16 x i16>
  %tmp2 = add <16 x i16> %tmp0, %tmp1
  ret <16 x i16> %tmp2
}

; COST-LABEL: uaddl2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i16> %a to <8 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <8 x i16> %b to <8 x i32>
; CODE-LABEL: uaddl2_4s
; CODE:       uaddl2 v2.4s, v0.8h, v1.8h
; CODE-NEXT:  uaddl v0.4s, v0.4h, v1.4h
define <8 x i32> @uaddl2_4s(<8 x i16> %a, <8 x i16> %b) {
  %tmp0 = zext <8 x i16> %a to <8 x i32>
  %tmp1 = zext <8 x i16> %b to <8 x i32>
  %tmp2 = add <8 x i32> %tmp0, %tmp1
  ret <8 x i32> %tmp2
}

; COST-LABEL: uaddl2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i32> %a to <4 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <4 x i32> %b to <4 x i64>
; CODE-LABEL: uaddl2_2d
; CODE:       uaddl2 v2.2d, v0.4s, v1.4s
; CODE-NEXT:  uaddl v0.2d, v0.2s, v1.2s
define <4 x i64> @uaddl2_2d(<4 x i32> %a, <4 x i32> %b) {
  %tmp0 = zext <4 x i32> %a to <4 x i64>
  %tmp1 = zext <4 x i32> %b to <4 x i64>
  %tmp2 = add <4 x i64> %tmp0, %tmp1
  ret <4 x i64> %tmp2
}

; COST-LABEL: saddl_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i8> %a to <8 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <8 x i8> %b to <8 x i16>
; CODE-LABEL: saddl_8h
; CODE:       saddl v0.8h, v0.8b, v1.8b
define <8 x i16> @saddl_8h(<8 x i8> %a, <8 x i8> %b) {
  %tmp0 = sext <8 x i8> %a to <8 x i16>
  %tmp1 = sext <8 x i8> %b to <8 x i16>
  %tmp2 = add <8 x i16> %tmp0, %tmp1
  ret <8 x i16> %tmp2
}

; COST-LABEL: saddl_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i16> %a to <4 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <4 x i16> %b to <4 x i32>
; CODE-LABEL: saddl_4s
; CODE:       saddl v0.4s, v0.4h, v1.4h
define <4 x i32> @saddl_4s(<4 x i16> %a, <4 x i16> %b) {
  %tmp0 = sext <4 x i16> %a to <4 x i32>
  %tmp1 = sext <4 x i16> %b to <4 x i32>
  %tmp2 = add <4 x i32> %tmp0, %tmp1
  ret <4 x i32> %tmp2
}

; COST-LABEL: saddl_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <2 x i32> %a to <2 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <2 x i32> %b to <2 x i64>
; CODE-LABEL: saddl_2d
; CODE:       saddl v0.2d, v0.2s, v1.2s
define <2 x i64> @saddl_2d(<2 x i32> %a, <2 x i32> %b) {
  %tmp0 = sext <2 x i32> %a to <2 x i64>
  %tmp1 = sext <2 x i32> %b to <2 x i64>
  %tmp2 = add <2 x i64> %tmp0, %tmp1
  ret <2 x i64> %tmp2
}

; COST-LABEL: saddl2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <16 x i8> %a to <16 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <16 x i8> %b to <16 x i16>
; CODE-LABEL: saddl2_8h
; CODE:       saddl2 v2.8h, v0.16b, v1.16b
; CODE-NEXT:  saddl v0.8h, v0.8b, v1.8b
define <16 x i16> @saddl2_8h(<16 x i8> %a, <16 x i8> %b) {
  %tmp0 = sext <16 x i8> %a to <16 x i16>
  %tmp1 = sext <16 x i8> %b to <16 x i16>
  %tmp2 = add <16 x i16> %tmp0, %tmp1
  ret <16 x i16> %tmp2
}

; COST-LABEL: saddl2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i16> %a to <8 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <8 x i16> %b to <8 x i32>
; CODE-LABEL: saddl2_4s
; CODE:       saddl2 v2.4s, v0.8h, v1.8h
; CODE-NEXT:  saddl v0.4s, v0.4h, v1.4h
define <8 x i32> @saddl2_4s(<8 x i16> %a, <8 x i16> %b) {
  %tmp0 = sext <8 x i16> %a to <8 x i32>
  %tmp1 = sext <8 x i16> %b to <8 x i32>
  %tmp2 = add <8 x i32> %tmp0, %tmp1
  ret <8 x i32> %tmp2
}

; COST-LABEL: saddl2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i32> %a to <4 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <4 x i32> %b to <4 x i64>
; CODE-LABEL: saddl2_2d
; CODE:       saddl2 v2.2d, v0.4s, v1.4s
; CODE-NEXT:  saddl v0.2d, v0.2s, v1.2s
define <4 x i64> @saddl2_2d(<4 x i32> %a, <4 x i32> %b) {
  %tmp0 = sext <4 x i32> %a to <4 x i64>
  %tmp1 = sext <4 x i32> %b to <4 x i64>
  %tmp2 = add <4 x i64> %tmp0, %tmp1
  ret <4 x i64> %tmp2
}

; COST-LABEL: usubl_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <8 x i8> %b to <8 x i16>
; CODE-LABEL: usubl_8h
; CODE:       usubl v0.8h, v0.8b, v1.8b
define <8 x i16> @usubl_8h(<8 x i8> %a, <8 x i8> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = zext <8 x i8> %b to <8 x i16>
  %tmp2 = sub <8 x i16> %tmp0, %tmp1
  ret <8 x i16> %tmp2
}

; COST-LABEL: usubl_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i16> %a to <4 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <4 x i16> %b to <4 x i32>
; CODE-LABEL: usubl_4s
; CODE:       usubl v0.4s, v0.4h, v1.4h
define <4 x i32> @usubl_4s(<4 x i16> %a, <4 x i16> %b) {
  %tmp0 = zext <4 x i16> %a to <4 x i32>
  %tmp1 = zext <4 x i16> %b to <4 x i32>
  %tmp2 = sub <4 x i32> %tmp0, %tmp1
  ret <4 x i32> %tmp2
}

; COST-LABEL: usubl_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <2 x i32> %a to <2 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <2 x i32> %b to <2 x i64>
; CODE-LABEL: usubl_2d
; CODE:       usubl v0.2d, v0.2s, v1.2s
define <2 x i64> @usubl_2d(<2 x i32> %a, <2 x i32> %b) {
  %tmp0 = zext <2 x i32> %a to <2 x i64>
  %tmp1 = zext <2 x i32> %b to <2 x i64>
  %tmp2 = sub <2 x i64> %tmp0, %tmp1
  ret <2 x i64> %tmp2
}

; COST-LABEL: usubl2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <16 x i8> %a to <16 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <16 x i8> %b to <16 x i16>
; CODE-LABEL: usubl2_8h
; CODE:       usubl2 v2.8h, v0.16b, v1.16b
; CODE-NEXT:  usubl v0.8h, v0.8b, v1.8b
define <16 x i16> @usubl2_8h(<16 x i8> %a, <16 x i8> %b) {
  %tmp0 = zext <16 x i8> %a to <16 x i16>
  %tmp1 = zext <16 x i8> %b to <16 x i16>
  %tmp2 = sub <16 x i16> %tmp0, %tmp1
  ret <16 x i16> %tmp2
}

; COST-LABEL: usubl2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i16> %a to <8 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <8 x i16> %b to <8 x i32>
; CODE-LABEL: usubl2_4s
; CODE:       usubl2 v2.4s, v0.8h, v1.8h
; CODE-NEXT:  usubl v0.4s, v0.4h, v1.4h
define <8 x i32> @usubl2_4s(<8 x i16> %a, <8 x i16> %b) {
  %tmp0 = zext <8 x i16> %a to <8 x i32>
  %tmp1 = zext <8 x i16> %b to <8 x i32>
  %tmp2 = sub <8 x i32> %tmp0, %tmp1
  ret <8 x i32> %tmp2
}

; COST-LABEL: usubl2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i32> %a to <4 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <4 x i32> %b to <4 x i64>
; CODE-LABEL: usubl2_2d
; CODE:       usubl2 v2.2d, v0.4s, v1.4s
; CODE-NEXT:  usubl v0.2d, v0.2s, v1.2s
define <4 x i64> @usubl2_2d(<4 x i32> %a, <4 x i32> %b) {
  %tmp0 = zext <4 x i32> %a to <4 x i64>
  %tmp1 = zext <4 x i32> %b to <4 x i64>
  %tmp2 = sub <4 x i64> %tmp0, %tmp1
  ret <4 x i64> %tmp2
}

; COST-LABEL: ssubl_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i8> %a to <8 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <8 x i8> %b to <8 x i16>
; CODE-LABEL: ssubl_8h
; CODE:       ssubl v0.8h, v0.8b, v1.8b
define <8 x i16> @ssubl_8h(<8 x i8> %a, <8 x i8> %b) {
  %tmp0 = sext <8 x i8> %a to <8 x i16>
  %tmp1 = sext <8 x i8> %b to <8 x i16>
  %tmp2 = sub <8 x i16> %tmp0, %tmp1
  ret <8 x i16> %tmp2
}

; COST-LABEL: ssubl_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i16> %a to <4 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <4 x i16> %b to <4 x i32>
; CODE-LABEL: ssubl_4s
; CODE:       ssubl v0.4s, v0.4h, v1.4h
define <4 x i32> @ssubl_4s(<4 x i16> %a, <4 x i16> %b) {
  %tmp0 = sext <4 x i16> %a to <4 x i32>
  %tmp1 = sext <4 x i16> %b to <4 x i32>
  %tmp2 = sub <4 x i32> %tmp0, %tmp1
  ret <4 x i32> %tmp2
}

; COST-LABEL: ssubl_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <2 x i32> %a to <2 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <2 x i32> %b to <2 x i64>
; CODE-LABEL: ssubl_2d
; CODE:       ssubl v0.2d, v0.2s, v1.2s
define <2 x i64> @ssubl_2d(<2 x i32> %a, <2 x i32> %b) {
  %tmp0 = sext <2 x i32> %a to <2 x i64>
  %tmp1 = sext <2 x i32> %b to <2 x i64>
  %tmp2 = sub <2 x i64> %tmp0, %tmp1
  ret <2 x i64> %tmp2
}

; COST-LABEL: ssubl2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <16 x i8> %a to <16 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <16 x i8> %b to <16 x i16>
; CODE-LABEL: ssubl2_8h
; CODE:       ssubl2 v2.8h, v0.16b, v1.16b
; CODE-NEXT:  ssubl v0.8h, v0.8b, v1.8b
define <16 x i16> @ssubl2_8h(<16 x i8> %a, <16 x i8> %b) {
  %tmp0 = sext <16 x i8> %a to <16 x i16>
  %tmp1 = sext <16 x i8> %b to <16 x i16>
  %tmp2 = sub <16 x i16> %tmp0, %tmp1
  ret <16 x i16> %tmp2
}

; COST-LABEL: ssubl2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i16> %a to <8 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <8 x i16> %b to <8 x i32>
; CODE-LABEL: ssubl2_4s
; CODE:       ssubl2 v2.4s, v0.8h, v1.8h
; CODE-NEXT:  ssubl v0.4s, v0.4h, v1.4h
define <8 x i32> @ssubl2_4s(<8 x i16> %a, <8 x i16> %b) {
  %tmp0 = sext <8 x i16> %a to <8 x i32>
  %tmp1 = sext <8 x i16> %b to <8 x i32>
  %tmp2 = sub <8 x i32> %tmp0, %tmp1
  ret <8 x i32> %tmp2
}

; COST-LABEL: ssubl2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i32> %a to <4 x i64>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = sext <4 x i32> %b to <4 x i64>
; CODE-LABEL: ssubl2_2d
; CODE:       ssubl2 v2.2d, v0.4s, v1.4s
; CODE-NEXT:  ssubl v0.2d, v0.2s, v1.2s
define <4 x i64> @ssubl2_2d(<4 x i32> %a, <4 x i32> %b) {
  %tmp0 = sext <4 x i32> %a to <4 x i64>
  %tmp1 = sext <4 x i32> %b to <4 x i64>
  %tmp2 = sub <4 x i64> %tmp0, %tmp1
  ret <4 x i64> %tmp2
}

; COST-LABEL: uaddw_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
; CODE-LABEL: uaddw_8h
; CODE:       uaddw v0.8h, v1.8h, v0.8b
define <8 x i16> @uaddw_8h(<8 x i8> %a, <8 x i16> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = add <8 x i16> %b, %tmp0
  ret <8 x i16> %tmp1
}

; COST-LABEL: uaddw_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i16> %a to <4 x i32>
; CODE-LABEL: uaddw_4s
; CODE:       uaddw v0.4s, v1.4s, v0.4h
define <4 x i32> @uaddw_4s(<4 x i16> %a, <4 x i32> %b) {
  %tmp0 = zext <4 x i16> %a to <4 x i32>
  %tmp1 = add <4 x i32> %b, %tmp0
  ret <4 x i32> %tmp1
}

; COST-LABEL: uaddw_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <2 x i32> %a to <2 x i64>
; CODE-LABEL: uaddw_2d
; CODE:       uaddw v0.2d, v1.2d, v0.2s
define <2 x i64> @uaddw_2d(<2 x i32> %a, <2 x i64> %b) {
  %tmp0 = zext <2 x i32> %a to <2 x i64>
  %tmp1 = add <2 x i64> %b, %tmp0
  ret <2 x i64> %tmp1
}

; COST-LABEL: uaddw2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <16 x i8> %a to <16 x i16>
; CODE-LABEL: uaddw2_8h
; CODE:       uaddw2 v2.8h, v2.8h, v0.16b
; CODE-NEXT:  uaddw v0.8h, v1.8h, v0.8b
define <16 x i16> @uaddw2_8h(<16 x i8> %a, <16 x i16> %b) {
  %tmp0 = zext <16 x i8> %a to <16 x i16>
  %tmp1 = add <16 x i16> %b, %tmp0
  ret <16 x i16> %tmp1
}

; COST-LABEL: uaddw2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i16> %a to <8 x i32>
; CODE-LABEL: uaddw2_4s
; CODE:       uaddw2 v2.4s, v2.4s, v0.8h
; CODE-NEXT:  uaddw v0.4s, v1.4s, v0.4h
define <8 x i32> @uaddw2_4s(<8 x i16> %a, <8 x i32> %b) {
  %tmp0 = zext <8 x i16> %a to <8 x i32>
  %tmp1 = add <8 x i32> %b, %tmp0
  ret <8 x i32> %tmp1
}

; COST-LABEL: uaddw2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i32> %a to <4 x i64>
; CODE-LABEL: uaddw2_2d
; CODE:       uaddw2 v2.2d, v2.2d, v0.4s
; CODE-NEXT:  uaddw v0.2d, v1.2d, v0.2s
define <4 x i64> @uaddw2_2d(<4 x i32> %a, <4 x i64> %b) {
  %tmp0 = zext <4 x i32> %a to <4 x i64>
  %tmp1 = add <4 x i64> %b, %tmp0
  ret <4 x i64> %tmp1
}

; COST-LABEL: saddw_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i8> %a to <8 x i16>
; CODE-LABEL: saddw_8h
; CODE:       saddw v0.8h, v1.8h, v0.8b
define <8 x i16> @saddw_8h(<8 x i8> %a, <8 x i16> %b) {
  %tmp0 = sext <8 x i8> %a to <8 x i16>
  %tmp1 = add <8 x i16> %b, %tmp0
  ret <8 x i16> %tmp1
}

; COST-LABEL: saddw_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i16> %a to <4 x i32>
; CODE-LABEL: saddw_4s
; CODE:       saddw v0.4s, v1.4s, v0.4h
define <4 x i32> @saddw_4s(<4 x i16> %a, <4 x i32> %b) {
  %tmp0 = sext <4 x i16> %a to <4 x i32>
  %tmp1 = add <4 x i32> %b, %tmp0
  ret <4 x i32> %tmp1
}

; COST-LABEL: saddw_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <2 x i32> %a to <2 x i64>
; CODE-LABEL: saddw_2d
; CODE:       saddw v0.2d, v1.2d, v0.2s
define <2 x i64> @saddw_2d(<2 x i32> %a, <2 x i64> %b) {
  %tmp0 = sext <2 x i32> %a to <2 x i64>
  %tmp1 = add <2 x i64> %b, %tmp0
  ret <2 x i64> %tmp1
}

; COST-LABEL: saddw2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <16 x i8> %a to <16 x i16>
; CODE-LABEL: saddw2_8h
; CODE:       saddw2 v2.8h, v2.8h, v0.16b
; CODE-NEXT:  saddw v0.8h, v1.8h, v0.8b
define <16 x i16> @saddw2_8h(<16 x i8> %a, <16 x i16> %b) {
  %tmp0 = sext <16 x i8> %a to <16 x i16>
  %tmp1 = add <16 x i16> %b, %tmp0
  ret <16 x i16> %tmp1
}

; COST-LABEL: saddw2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i16> %a to <8 x i32>
; CODE-LABEL: saddw2_4s
; CODE:       saddw2 v2.4s, v2.4s, v0.8h
; CODE-NEXT:  saddw v0.4s, v1.4s, v0.4h
define <8 x i32> @saddw2_4s(<8 x i16> %a, <8 x i32> %b) {
  %tmp0 = sext <8 x i16> %a to <8 x i32>
  %tmp1 = add <8 x i32> %b, %tmp0
  ret <8 x i32> %tmp1
}

; COST-LABEL: saddw2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i32> %a to <4 x i64>
; CODE-LABEL: saddw2_2d
; CODE:       saddw2 v2.2d, v2.2d, v0.4s
; CODE-NEXT:  saddw v0.2d, v1.2d, v0.2s
define <4 x i64> @saddw2_2d(<4 x i32> %a, <4 x i64> %b) {
  %tmp0 = sext <4 x i32> %a to <4 x i64>
  %tmp1 = add <4 x i64> %b, %tmp0
  ret <4 x i64> %tmp1
}

; COST-LABEL: usubw_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
; CODE-LABEL: usubw_8h
; CODE:       usubw v0.8h, v1.8h, v0.8b
define <8 x i16> @usubw_8h(<8 x i8> %a, <8 x i16> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = sub <8 x i16> %b, %tmp0
  ret <8 x i16> %tmp1
}

; COST-LABEL: usubw_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i16> %a to <4 x i32>
; CODE-LABEL: usubw_4s
; CODE:       usubw v0.4s, v1.4s, v0.4h
define <4 x i32> @usubw_4s(<4 x i16> %a, <4 x i32> %b) {
  %tmp0 = zext <4 x i16> %a to <4 x i32>
  %tmp1 = sub <4 x i32> %b, %tmp0
  ret <4 x i32> %tmp1
}

; COST-LABEL: usubw_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <2 x i32> %a to <2 x i64>
; CODE-LABEL: usubw_2d
; CODE:       usubw v0.2d, v1.2d, v0.2s
define <2 x i64> @usubw_2d(<2 x i32> %a, <2 x i64> %b) {
  %tmp0 = zext <2 x i32> %a to <2 x i64>
  %tmp1 = sub <2 x i64> %b, %tmp0
  ret <2 x i64> %tmp1
}

; COST-LABEL: usubw2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <16 x i8> %a to <16 x i16>
; CODE-LABEL: usubw2_8h
; CODE:       usubw2 v2.8h, v2.8h, v0.16b
; CODE-NEXT:  usubw v0.8h, v1.8h, v0.8b
define <16 x i16> @usubw2_8h(<16 x i8> %a, <16 x i16> %b) {
  %tmp0 = zext <16 x i8> %a to <16 x i16>
  %tmp1 = sub <16 x i16> %b, %tmp0
  ret <16 x i16> %tmp1
}

; COST-LABEL: usubw2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i16> %a to <8 x i32>
; CODE-LABEL: usubw2_4s
; CODE:       usubw2 v2.4s, v2.4s, v0.8h
; CODE-NEXT:  usubw v0.4s, v1.4s, v0.4h
define <8 x i32> @usubw2_4s(<8 x i16> %a, <8 x i32> %b) {
  %tmp0 = zext <8 x i16> %a to <8 x i32>
  %tmp1 = sub <8 x i32> %b, %tmp0
  ret <8 x i32> %tmp1
}

; COST-LABEL: usubw2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <4 x i32> %a to <4 x i64>
; CODE-LABEL: usubw2_2d
; CODE:       usubw2 v2.2d, v2.2d, v0.4s
; CODE-NEXT:  usubw v0.2d, v1.2d, v0.2s
define <4 x i64> @usubw2_2d(<4 x i32> %a, <4 x i64> %b) {
  %tmp0 = zext <4 x i32> %a to <4 x i64>
  %tmp1 = sub <4 x i64> %b, %tmp0
  ret <4 x i64> %tmp1
}

; COST-LABEL: ssubw_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i8> %a to <8 x i16>
; CODE-LABEL: ssubw_8h
; CODE:       ssubw v0.8h, v1.8h, v0.8b
define <8 x i16> @ssubw_8h(<8 x i8> %a, <8 x i16> %b) {
  %tmp0 = sext <8 x i8> %a to <8 x i16>
  %tmp1 = sub <8 x i16> %b, %tmp0
  ret <8 x i16> %tmp1
}

; COST-LABEL: ssubw_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i16> %a to <4 x i32>
; CODE-LABEL: ssubw_4s
; CODE:       ssubw v0.4s, v1.4s, v0.4h
define <4 x i32> @ssubw_4s(<4 x i16> %a, <4 x i32> %b) {
  %tmp0 = sext <4 x i16> %a to <4 x i32>
  %tmp1 = sub <4 x i32> %b, %tmp0
  ret <4 x i32> %tmp1
}

; COST-LABEL: ssubw_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <2 x i32> %a to <2 x i64>
; CODE-LABEL: ssubw_2d
; CODE:       ssubw v0.2d, v1.2d, v0.2s
define <2 x i64> @ssubw_2d(<2 x i32> %a, <2 x i64> %b) {
  %tmp0 = sext <2 x i32> %a to <2 x i64>
  %tmp1 = sub <2 x i64> %b, %tmp0
  ret <2 x i64> %tmp1
}

; COST-LABEL: ssubw2_8h
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <16 x i8> %a to <16 x i16>
; CODE-LABEL: ssubw2_8h
; CODE:       ssubw2 v2.8h, v2.8h, v0.16b
; CODE-NEXT:  ssubw v0.8h, v1.8h, v0.8b
define <16 x i16> @ssubw2_8h(<16 x i8> %a, <16 x i16> %b) {
  %tmp0 = sext <16 x i8> %a to <16 x i16>
  %tmp1 = sub <16 x i16> %b, %tmp0
  ret <16 x i16> %tmp1
}

; COST-LABEL: ssubw2_4s
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <8 x i16> %a to <8 x i32>
; CODE-LABEL: ssubw2_4s
; CODE:       ssubw2 v2.4s, v2.4s, v0.8h
; CODE-NEXT:  ssubw v0.4s, v1.4s, v0.4h
define <8 x i32> @ssubw2_4s(<8 x i16> %a, <8 x i32> %b) {
  %tmp0 = sext <8 x i16> %a to <8 x i32>
  %tmp1 = sub <8 x i32> %b, %tmp0
  ret <8 x i32> %tmp1
}

; COST-LABEL: ssubw2_2d
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = sext <4 x i32> %a to <4 x i64>
; CODE-LABEL: ssubw2_2d
; CODE:       ssubw2 v2.2d, v2.2d, v0.4s
; CODE-NEXT:  ssubw v0.2d, v1.2d, v0.2s
define <4 x i64> @ssubw2_2d(<4 x i32> %a, <4 x i64> %b) {
  %tmp0 = sext <4 x i32> %a to <4 x i64>
  %tmp1 = sub <4 x i64> %b, %tmp0
  ret <4 x i64> %tmp1
}

; COST-LABEL: neg_wrong_operand_order
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
define <8 x i16> @neg_wrong_operand_order(<8 x i8> %a, <8 x i16> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = sub <8 x i16> %tmp0, %b
  ret <8 x i16> %tmp1
}

; COST-LABEL: neg_non_widening_op
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
define <8 x i16> @neg_non_widening_op(<8 x i8> %a, <8 x i16> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = udiv <8 x i16> %b, %tmp0
  ret <8 x i16> %tmp1
}

; COST-LABEL: neg_dissimilar_operand_kind_0
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = sext <8 x i8> %a to <8 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <8 x i8> %b to <8 x i16>
define <8 x i16> @neg_dissimilar_operand_kind_0(<8 x i8> %a, <8 x i8> %b) {
  %tmp0 = sext <8 x i8> %a to <8 x i16>
  %tmp1 = zext <8 x i8> %b to <8 x i16>
  %tmp2 = add <8 x i16> %tmp0, %tmp1
  ret <8 x i16> %tmp2
}

; COST-LABEL: neg_dissimilar_operand_kind_1
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <4 x i8> %a to <4 x i32>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <4 x i16> %b to <4 x i32>
define <4 x i32> @neg_dissimilar_operand_kind_1(<4 x i8> %a, <4 x i16> %b) {
  %tmp0 = zext <4 x i8> %a to <4 x i32>
  %tmp1 = zext <4 x i16> %b to <4 x i32>
  %tmp2 = add <4 x i32> %tmp0, %tmp1
  ret <4 x i32> %tmp2
}

; COST-LABEL: neg_illegal_vector_type_0
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <16 x i4> %a to <16 x i8>
define <16 x i8> @neg_illegal_vector_type_0(<16 x i4> %a, <16 x i8> %b) {
  %tmp0 = zext <16 x i4> %a to <16 x i8>
  %tmp1 = sub <16 x i8> %b, %tmp0
  ret <16 x i8> %tmp1
}

; COST-LABEL: neg_llegal_vector_type_1
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <1 x i16> %a to <1 x i32>
define <1 x i32> @neg_llegal_vector_type_1(<1 x i16> %a, <1 x i32> %b) {
  %tmp0 = zext <1 x i16> %a to <1 x i32>
  %tmp1 = add <1 x i32> %b, %tmp0
  ret <1 x i32> %tmp1
}

; COST-LABEL: neg_llegal_vector_type_2
; COST-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %tmp0 = zext <4 x i16> %a to <4 x i64>
define <4 x i64> @neg_llegal_vector_type_2(<4 x i16> %a, <4 x i64> %b) {
  %tmp0 = zext <4 x i16> %a to <4 x i64>
  %tmp1 = add <4 x i64> %b, %tmp0
  ret <4 x i64> %tmp1
}

; COST-LABEL: neg_llegal_vector_type_3
; COST-NEXT:  Cost Model: Found an estimated cost of 9 for instruction: %tmp0 = zext <3 x i34> %a to <3 x i68>
define <3 x i68> @neg_llegal_vector_type_3(<3 x i34> %a, <3 x i68> %b) {
  %tmp0 = zext <3 x i34> %a to <3 x i68>
  %tmp1 = add <3 x i68> %b, %tmp0
  ret <3 x i68> %tmp1
}


; COST-LABEL: usubl_abs_diff_multi_user
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp1 = zext <8 x i8> %b to <8 x i16>
define <8 x i16> @usubl_abs_diff_multi_user(<8 x i8> %a, <8 x i8> %b) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = zext <8 x i8> %b to <8 x i16>
  %cmp = icmp ugt <8 x i8> %a, %b
  %sub1 = sub <8 x i16> %tmp0, %tmp1
  %sub2 = sub <8 x i16> %tmp1, %tmp0
  %abs = select <8 x i1> %cmp, <8 x i16> %sub1, <8 x i16> %sub2
  ret <8 x i16> %abs
}

; COST-LABEL: uaddw_multi_user
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
define <8 x i16> @uaddw_multi_user(<8 x i8> %a, <8 x i16> %b, <8 x i16> %c) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = add <8 x i16> %b, %tmp0
  %tmp2 = add <8 x i16> %c, %tmp0
  %tmp3 = add <8 x i16> %tmp1, %tmp2
  ret <8 x i16> %tmp3
}

; COST-LABEL: usubw_multi_user
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
define <8 x i16> @usubw_multi_user(<8 x i8> %a, <8 x i16> %b, <8 x i16> %c, i1 %flag) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = sub <8 x i16> %b, %tmp0
  %tmp2 = sub <8 x i16> %c, %tmp0
  %tmp3 = select i1 %flag, <8 x i16> %tmp1, <8 x i16> %tmp2
  ret <8 x i16> %tmp3
}

; COST-LABEL: icmp_v2i64
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %c = icmp sgt <2 x i64> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %s = sext <2 x i1> %c to <2 x i64>
; CODE-LABEL: icmp_v2i64
; CODE:       cmgt v0.2d, v0.2d, v1.2d
; CODE-NEXT:  ret
define <2 x i64> @icmp_v2i64(<2 x i64> %a, <2 x i64> %b) {
  %c = icmp sgt <2 x i64> %a, %b
  %s = sext <2 x i1> %c to <2 x i64>
  ret <2 x i64> %s
}

; COST-LABEL: fcmp_v4f32
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %c = fcmp olt <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %s = sext <4 x i1> %c to <4 x i32>
; CODE-LABEL: fcmp_v4f32
; CODE:       fcmgt v0.4s, v1.4s, v0.4s
; CODE-NEXT:  ret
define <4 x i32> @fcmp_v4f32(<4 x float> %a, <4 x float> %b) {
  %c = fcmp olt <4 x float> %a, %b
  %s = sext <4 x i1> %c to <4 x i32>
  ret <4 x i32> %s
}

; COST-LABEL: icmp_v8i64
; COST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %c = icmp sgt <8 x i64> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %s = sext <8 x i1> %c to <8 x i64>
; CODE-LABEL: icmp_v8i64
; CODE:       cmgt v2.2d, v2.2d, v6.2d
; CODE-NEXT:  cmgt v0.2d, v0.2d, v4.2d
; CODE-NEXT:  cmgt v1.2d, v1.2d, v5.2d
; CODE-NEXT:  cmgt v3.2d, v3.2d, v7.2d
; CODE-NEXT:  ret
define <8 x i64> @icmp_v8i64(<8 x i64> %a, <8 x i64> %b) {
  %c = icmp sgt <8 x i64> %a, %b
  %s = sext <8 x i1> %c to <8 x i64>
  ret <8 x i64> %s
}

; COST-LABEL: fcmp_v4f16
; COST-NEXT:  Cost Model: Found an estimated cost of 4 for instruction: %c = fcmp olt <4 x half> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %s = sext <4 x i1> %c to <4 x i16>
; CODE-LABEL: fcmp_v4f16
; CODE:       fcvtl v0.4s, v0.4h
; CODE-NEXT:  fcvtl v1.4s, v1.4h
; CODE-NEXT:  fcmgt v0.4s, v1.4s, v0.4s
; CODE-NEXT:  xtn v0.4h, v0.4s
; CODE-NEXT:  ret
define <4 x i16> @fcmp_v4f16(<4 x half> %a, <4 x half> %b) {
  %c = fcmp olt <4 x half> %a, %b
  %s = sext <4 x i1> %c to <4 x i16>
  ret <4 x i16> %s
}

; COST-LABEL: icmp_v2i128
; COST-NEXT:  Cost Model: Found an estimated cost of 8 for instruction: %c = icmp sgt <2 x i128> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 6 for instruction: %s = sext <2 x i1> %c to <2 x i128>
; CODE-LABEL: icmp_v2i128
; CODE:       cmp x4, x0
; CODE-NEXT:  sbcs xzr, x5, x1
; CODE-NEXT:  csetm x0, lt
; CODE-NEXT:  cmp x6, x2
; CODE-NEXT:  sbcs xzr, x7, x3
; CODE-NEXT:  mov x1, x0
; CODE-NEXT:  csetm x2, lt
; CODE-NEXT:  mov x3, x2
; CODE-NEXT:  ret
define <2 x i128> @icmp_v2i128(<2 x i128> %a, <2 x i128> %b) {
  %c = icmp sgt <2 x i128> %a, %b
  %s = sext <2 x i1> %c to <2 x i128>
  ret <2 x i128> %s
}

; COST-LABEL: fcmp_v2f128
; COST-NEXT:  Cost Model: Found an estimated cost of 20 for instruction: %c = fcmp olt <2 x fp128> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 6 for instruction: %s = sext <2 x i1> %c to <2 x i128>
; CODE-LABEL: fcmp_v2f128
; CODE:       bl __lttf2
; CODE:       csetm x19, mi
; CODE-NEXT:  bl __lttf2
; CODE:       csetm x2, mi
define <2 x i128> @fcmp_v2f128(<2 x fp128> %a, <2 x fp128> %b) {
  %c = fcmp olt <2 x fp128> %a, %b
  %s = sext <2 x i1> %c to <2 x i128>
  ret <2 x i128> %s
}

; COST-LABEL: neg_add_multi_user_not_all_absorbed
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
define <8 x i16> @neg_add_multi_user_not_all_absorbed(<8 x i8> %a, <8 x i16> %b, <8 x i16> %c) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = add <8 x i16> %b, %tmp0
  %tmp2 = udiv <8 x i16> %c, %tmp0
  %tmp3 = add <8 x i16> %tmp1, %tmp2
  ret <8 x i16> %tmp3
}

; COST-LABEL: neg_sub_multi_user_not_all_absorbed
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %tmp0 = zext <8 x i8> %a to <8 x i16>
define <8 x i16> @neg_sub_multi_user_not_all_absorbed(<8 x i8> %a, <8 x i16> %b, <8 x i16> %c, i1 %flag) {
  %tmp0 = zext <8 x i8> %a to <8 x i16>
  %tmp1 = sub <8 x i16> %b, %tmp0
  %tmp2 = udiv <8 x i16> %c, %tmp0
  %tmp3 = select i1 %flag, <8 x i16> %tmp1, <8 x i16> %tmp2
  ret <8 x i16> %tmp3
}

; COST-LABEL: neg_icmp_v8i8_to_v8i16
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %c = icmp ule <8 x i8> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %s = sext <8 x i1> %c to <8 x i16>
; CODE-LABEL: neg_icmp_v8i8_to_v8i16
; CODE:       cmhs v0.8b, v1.8b, v0.8b
; CODE-NEXT:  sshll v0.8h, v0.8b, #0
define <8 x i16> @neg_icmp_v8i8_to_v8i16(<8 x i8> %a, <8 x i8> %b) {
  %c = icmp ule <8 x i8> %a, %b
  %s = sext <8 x i1> %c to <8 x i16>
  ret <8 x i16> %s
}

; COST-LABEL: neg_scalar_icmp_i32
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %c = icmp eq i32 %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %s = sext i1 %c to i32
; CODE-LABEL: neg_scalar_icmp_i32
; CODE:       cmp w0, w1
; CODE-NEXT:  csetm w0, eq
define i32 @neg_scalar_icmp_i32(i32 %a, i32 %b) {
  %c = icmp eq i32 %a, %b
  %s = sext i1 %c to i32
  ret i32 %s
}

; COST-LABEL: neg_icmp_v4i32_other_block
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %c = icmp eq <4 x i32> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: br i1 %f, label %t, label %e
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %s = sext <4 x i1> %c to <4 x i32>
; CODE-LABEL: neg_icmp_v4i32_other_block
; CODE:       cmeq v0.4s, v0.4s, v1.4s
; CODE:       csetm w8, ne
; CODE-NEXT:  dup v1.4s, w8
; CODE-NEXT:  and v0.16b, v0.16b, v1.16b
define <4 x i32> @neg_icmp_v4i32_other_block(<4 x i32> %a, <4 x i32> %b, i1 %f) {
entry:
  %c = icmp eq <4 x i32> %a, %b
  br i1 %f, label %t, label %e
t:
  %s = sext <4 x i1> %c to <4 x i32>
  ret <4 x i32> %s
e:
  ret <4 x i32> zeroinitializer
}
