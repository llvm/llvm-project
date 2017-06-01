; RUN: llc -mattr=addsubiw < %s -march=avr | FileCheck %s

define i8 @add8_reg_reg(i8 %a, i8 %b) {
; CHECK-LABEL: add8_reg_reg:
; CHECK: add r24, r22
    %result = add i8 %a, %b
    ret i8 %result
}

define i8 @add8_reg_imm(i8 %a) {
; CHECK-LABEL: add8_reg_imm:
; CHECK: subi r24, -5
    %result = add i8 %a, 5
    ret i8 %result
}

define i8 @add8_reg_increment(i8 %a) {
; CHECK-LABEL: add8_reg_increment:
; CHECK: inc r24
    %result = add i8 %a, 1
    ret i8 %result
}


define i16 @add16_reg_reg(i16 %a, i16 %b) {
; CHECK-LABEL: add16_reg_reg:
; CHECK: add r24, r22
; CHECK: adc r25, r23
    %result = add i16 %a, %b
    ret i16 %result
}

define i16 @add16_reg_imm(i16 %a) {
; CHECK-LABEL: add16_reg_imm:
; CHECK: adiw r24, 63
    %result = add i16 %a, 63
    ret i16 %result
}

define i16 @add16_reg_imm_subi(i16 %a) {
; CHECK-LABEL: add16_reg_imm_subi:
; CHECK: subi r24, 133
; CHECK: sbci r25, 255
    %result = add i16 %a, 123
    ret i16 %result
}

define i32 @add32_reg_reg(i32 %a, i32 %b) {
; CHECK-LABEL: add32_reg_reg:
; CHECK: add r22, r18
; CHECK: adc r23, r19
; CHECK: adc r24, r20
; CHECK: adc r25, r21
    %result = add i32 %a, %b
    ret i32 %result
}

define i32 @add32_reg_imm(i32 %a) {
; CHECK-LABEL: add32_reg_imm:
; CHECK: subi r22, 251
; CHECK: sbci r23, 255
; CHECK: sbci r24, 255
; CHECK: sbci r25, 255
    %result = add i32 %a, 5
    ret i32 %result
}

define i64 @add64_reg_reg(i64 %a, i64 %b) {
; CHECK-LABEL: add64_reg_reg:
; CHECK: add r18, r10
; CHECK: adc r20, r12
; CHECK: adc r21, r13
; CHECK: adc r22, r14
; CHECK: adc r23, r15
; CHECK: adc r24, r16
; CHECK: adc r25, r17
    %result = add i64 %a, %b
    ret i64 %result
}

define i64 @add64_reg_imm(i64 %a) {
; CHECK-LABEL: add64_reg_imm:
; CHECK: subi r18, 251
; CHECK: sbci r19, 255
; CHECK: sbci r20, 255
; CHECK: sbci r21, 255
; CHECK: sbci r22, 255
; CHECK: sbci r23, 255
; CHECK: sbci r24, 255
; CHECK: sbci r25, 255
    %result = add i64 %a, 5
    ret i64 %result
}
