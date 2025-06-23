; RUN: llc < %s -mtriple=avr | FileCheck %s

define i8 @sub8_reg_reg(i8 %a, i8 %b) {
; CHECK-LABEL: sub8_reg_reg:
; CHECK: sub r24, r22
    %result = sub i8 %a, %b
    ret i8 %result
}

define i8 @sub8_reg_imm(i8 %a) {
; CHECK-LABEL: sub8_reg_imm:
; CHECK: subi r24, 5
    %result = sub i8 %a, 5
    ret i8 %result
}

define i8 @sub8_reg_decrement(i8 %a) {
; CHECK-LABEL: sub8_reg_decrement:
; CHECK: dec r24
    %result = sub i8 %a, 1
    ret i8 %result
}

define i16 @sub16_reg_reg(i16 %a, i16 %b) {
; CHECK-LABEL: sub16_reg_reg:
; CHECK: sub r24, r22
; CHECK: sbc r25, r23
    %result = sub i16 %a, %b
    ret i16 %result
}

define i16 @sub16_reg_imm(i16 %a) {
; CHECK-LABEL: sub16_reg_imm:
; CHECK: sbiw r24, 63
    %result = sub i16 %a, 63
    ret i16 %result
}

define i16 @sub16_reg_imm_subi(i16 %a) {
; CHECK-LABEL: sub16_reg_imm_subi:
; CHECK: subi r24, 210
; CHECK: sbci r25, 4
    %result = sub i16 %a, 1234
    ret i16 %result
}

define i32 @sub32_reg_reg(i32 %a, i32 %b) {
; CHECK-LABEL: sub32_reg_reg:
; CHECK: sub r22, r18
; CHECK: sbc r23, r19
; CHECK: sbc r24, r20
; CHECK: sbc r25, r21
    %result = sub i32 %a, %b
    ret i32 %result
}

define i32 @sub32_reg_imm(i32 %a) {
; CHECK-LABEL: sub32_reg_imm:
; CHECK: subi r22, 21
; CHECK: sbci r23, 205
; CHECK: sbci r24, 91
; CHECK: sbci r25, 7
    %result = sub i32 %a, 123456789
    ret i32 %result
}

define i64 @sub64_reg_reg(i64 %a, i64 %b) {
; CHECK-LABEL: sub64_reg_reg:
; CHECK: sub r18, r10
; CHECK: sbc r20, r12
; CHECK: sbc r21, r13
; CHECK: sbc r22, r14
; CHECK: sbc r23, r15
; CHECK: sbc r24, r16
; CHECK: sbc r25, r17
    %result = sub i64 %a, %b
    ret i64 %result
}

define i64 @sub64_reg_imm(i64 %a) {
; CHECK-LABEL: sub64_reg_imm:
; CHECK: subi r18, 204
; CHECK: sbci r19, 204
; CHECK: sbci r20, 104
; CHECK: sbci r21, 37
; CHECK: sbci r22, 25
; CHECK: sbci r23, 22
; CHECK: sbci r24, 236
; CHECK: sbci r25, 190
    %result = sub i64 %a, 13757395258967641292
    ret i64 %result
}
