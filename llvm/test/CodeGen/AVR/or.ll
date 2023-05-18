; RUN: llc < %s -march=avr | FileCheck %s

define i8 @or8_reg_reg(i8 %a, i8 %b) {
; CHECK-LABEL: or8_reg_reg:
; CHECK: or r24, r22
    %result = or i8 %a, %b
    ret i8 %result
}

define i8 @or8_reg_imm(i8 %a) {
; CHECK-LABEL: or8_reg_imm:
; CHECK: ori r24, 5
    %result = or i8 %a, 5
    ret i8 %result
}

define i16 @or16_reg_reg(i16 %a, i16 %b) {
; CHECK-LABEL: or16_reg_reg:
; CHECK: or r24, r22
; CHECK: or r25, r23
    %result = or i16 %a, %b
    ret i16 %result
}

define i16 @or16_reg_imm(i16 %a) {
; CHECK-LABEL: or16_reg_imm:
; CHECK: ori r24, 210
; CHECK: ori r25, 4
    %result = or i16 %a, 1234
    ret i16 %result
}

define i32 @or32_reg_reg(i32 %a, i32 %b) {
; CHECK-LABEL: or32_reg_reg:
; CHECK: or r22, r18
; CHECK: or r23, r19
; CHECK: or r24, r20
; CHECK: or r25, r21
    %result = or i32 %a, %b
    ret i32 %result
}

define i32 @or32_reg_imm(i32 %a) {
; CHECK-LABEL: or32_reg_imm:
; CHECK: ori r22, 21
; CHECK: ori r23, 205
; CHECK: ori r24, 91
; CHECK: ori r25, 7
    %result = or i32 %a, 123456789
    ret i32 %result
}

define i64 @or64_reg_reg(i64 %a, i64 %b) {
; CHECK-LABEL: or64_reg_reg:
; CHECK: or r18, r10
; CHECK: or r19, r11
; CHECK: or r20, r12
; CHECK: or r21, r13
; CHECK: or r22, r14
; CHECK: or r23, r15
; CHECK: or r24, r16
; CHECK: or r25, r17
    %result = or i64 %a, %b
    ret i64 %result
}

define i64 @or64_reg_imm(i64 %a) {
; CHECK-LABEL: or64_reg_imm:
; CHECK: ori r18, 204
; CHECK: ori r19, 204
; CHECK: ori r20, 204
; CHECK: ori r21, 204
; CHECK: ori r22, 204
; CHECK: ori r23, 204
; CHECK: ori r24, 204
; CHECK: ori r25, 204
    %result = or i64 %a, 14757395258967641292
    ret i64 %result
}

