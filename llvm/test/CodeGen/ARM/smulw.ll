; RUN: llc -mtriple=arm--none-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb--none-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s

; We cannot codegen the smulw[bt] or smlaw[bt] instructions for these functions,
; as the top 16 bits of the result would differ

define i32 @f1(i32 %a, i16 %b) {
; CHECK-LABEL: f1:
; CHECK: mul
; CHECK: asr
  %tmp1 = sext i16 %b to i32
  %tmp2 = mul i32 %a, %tmp1
  %tmp3 = ashr i32 %tmp2, 16
  ret i32 %tmp3
}

define i32 @f2(i32 %a, i16 %b, i32 %c) {
; CHECK-LABEL: f2:
; CHECK: mul
; CHECK: add{{.*}}, asr #16
  %tmp1 = sext i16 %b to i32
  %tmp2 = mul i32 %a, %tmp1
  %tmp3 = ashr i32 %tmp2, 16
  %tmp4 = add i32 %tmp3, %c
  ret i32 %tmp4
}
