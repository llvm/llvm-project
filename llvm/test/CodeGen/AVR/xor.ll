; RUN: llc < %s -mtriple=avr | FileCheck %s

define i8 @xor8_reg_reg(i8 %a, i8 %b) {
; CHECK-LABEL: xor8_reg_reg:
; CHECK: eor r24, r22
    %result = xor i8 %a, %b
    ret i8 %result
}

define i16 @xor16_reg_reg(i16 %a, i16 %b) {
; CHECK-LABEL: xor16_reg_reg:
; CHECK: eor r24, r22
; CHECK: eor r25, r23
    %result = xor i16 %a, %b
    ret i16 %result
}

define i32 @xor32_reg_reg(i32 %a, i32 %b) {
; CHECK-LABEL: xor32_reg_reg:
; CHECK: eor r22, r18
; CHECK: eor r23, r19
; CHECK: eor r24, r20
; CHECK: eor r25, r21
    %result = xor i32 %a, %b
    ret i32 %result
}

define i64 @xor64_reg_reg(i64 %a, i64 %b) {
; CHECK-LABEL: xor64_reg_reg:
; CHECK: eor r18, r10
; CHECK: eor r19, r11
; CHECK: eor r20, r12
; CHECK: eor r21, r13
; CHECK: eor r22, r14
; CHECK: eor r23, r15
; CHECK: eor r24, r16
; CHECK: eor r25, r17
    %result = xor i64 %a, %b
    ret i64 %result
}

