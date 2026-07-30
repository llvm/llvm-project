; RUN: llc < %s -mtriple=avr | FileCheck %s

define i8 @and8_reg_reg(i8 %a, i8 %b) {
; CHECK-LABEL: and8_reg_reg:
; CHECK: and r24, r22
    %result = and i8 %a, %b
    ret i8 %result
}

define i8 @and8_reg_imm(i8 %a) {
; CHECK-LABEL: and8_reg_imm:
; CHECK: andi r24, 5
    %result = and i8 %a, 5
    ret i8 %result
}

define i16 @and16_reg_reg(i16 %a, i16 %b) {
; CHECK-LABEL: and16_reg_reg:
; CHECK: and r24, r22
; CHECK: and r25, r23
    %result = and i16 %a, %b
    ret i16 %result
}

define i16 @and16_reg_imm(i16 %a) {
; CHECK-LABEL: and16_reg_imm:
; CHECK: andi r24, 210
; CHECK: andi r25, 4
    %result = and i16 %a, 1234
    ret i16 %result
}

define i32 @and32_reg_reg(i32 %a, i32 %b) {
; CHECK-LABEL: and32_reg_reg:
; CHECK: and r22, r18
; CHECK: and r23, r19
; CHECK: and r24, r20
; CHECK: and r25, r21
    %result = and i32 %a, %b
    ret i32 %result
}

define i32 @and32_reg_imm(i32 %a) {
; CHECK-LABEL: and32_reg_imm:
; CHECK: andi r22, 21
; CHECK: andi r23, 205
; CHECK: andi r24, 91
; CHECK: andi r25, 7
    %result = and i32 %a, 123456789
    ret i32 %result
}

define i64 @and64_reg_reg(i64 %a, i64 %b) {
; CHECK-LABEL: and64_reg_reg:
; CHECK: and r18, r10
; CHECK: and r19, r11
; CHECK: and r20, r12
; CHECK: and r21, r13
; CHECK: and r22, r14
; CHECK: and r23, r15
; CHECK: and r24, r16
; CHECK: and r25, r17
    %result = and i64 %a, %b
    ret i64 %result
}

define i64 @and64_reg_imm(i64 %a) {
; CHECK-LABEL: and64_reg_imm:
; CHECK: andi r18, 253
; Per PR 31345, we optimize away ANDI Rd, 0xff
; CHECK-NOT: andi r19, 255
; CHECK: andi r20, 155
; CHECK: andi r21, 88
; CHECK: andi r22, 76
; CHECK: andi r23, 73
; CHECK: andi r24, 31
; CHECK: andi r25, 242
    %result = and i64 %a, 17446744073709551613
    ret i64 %result
}

