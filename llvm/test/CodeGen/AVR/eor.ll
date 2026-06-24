; RUN: llc < %s -mtriple=avr | FileCheck %s

; Tests for the exclusive OR operation.

define i8 @eor8_reg_reg(i8 %a, i8 %b) {
; CHECK-LABEL: eor8_reg_reg:
; CHECK: eor r24, r22
    %result = xor i8 %a, %b
    ret i8 %result
}

define i8 @eor8_reg_imm(i8 %a) {
; CHECK-LABEL: eor8_reg_imm:
; CHECK: ldi r25, 5
; CHECK: eor r24, r25
    %result = xor i8 %a, 5
    ret i8 %result
}

define i16 @eor16_reg_reg(i16 %a, i16 %b) {
; CHECK-LABEL: eor16_reg_reg:
; CHECK: eor r24, r22
; CHECK: eor r25, r23
    %result = xor i16 %a, %b
    ret i16 %result
}

define i16 @eor16_reg_imm(i16 %a) {
; CHECK-LABEL: eor16_reg_imm:
; CHECK: ldi r18, 210
; CHECK: ldi r19, 4
; CHECK: eor r24, r18
; CHECK: eor r25, r19
    %result = xor i16 %a, 1234
    ret i16 %result
}

define i32 @eor32_reg_reg(i32 %a, i32 %b) {
; CHECK-LABEL: eor32_reg_reg:
; CHECK: eor r22, r18
; CHECK: eor r23, r19
; CHECK: eor r24, r20
; CHECK: eor r25, r21
    %result = xor i32 %a, %b
    ret i32 %result
}

define i32 @eor32_reg_imm(i32 %a) {
; CHECK-LABEL: eor32_reg_imm:
; CHECK: ldi r18, 210
; CHECK: ldi r19, 4
; CHECK: eor r22, r18
; CHECK: eor r23, r19
    %result = xor i32 %a, 1234
    ret i32 %result
}

define i64 @eor64_reg_reg(i64 %a, i64 %b) {
; CHECK-LABEL: eor64_reg_reg:
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

define i64 @eor64_reg_imm(i64 %a) {
; CHECK-LABEL: eor64_reg_imm:
; CHECK: ldi r30, 253
; CHECK: ldi r31, 255
; CHECK: eor r18, r30
; CHECK: eor r19, r31
; CHECK: ldi r30, 155
; CHECK: ldi r31, 88
; CHECK: eor r20, r30
; CHECK: eor r21, r31
; CHECK: ldi r30, 76
; CHECK: ldi r31, 73
; CHECK: eor r22, r30
; CHECK: eor r23, r31
; CHECK: ldi r30, 31
; CHECK: ldi r31, 242
; CHECK: eor r24, r30
; CHECK: eor r25, r31
    %result = xor i64 %a, 17446744073709551613
    ret i64 %result
}
