; RUN: llc < %s -mtriple=avr | FileCheck %s

; Unit test for: PR 31344

define i16 @or16_reg_imm_0xff00(i16 %a) {
; CHECK-LABEL: or16_reg_imm_0xff00
; CHECK-NOT: ori {{r[0-9]+}}, 0
; CHECK: ori {{r[0-9]+}}, 255
    %result = or i16 %a, 65280
    ret i16 %result
}

define i16 @or16_reg_imm_0xffb3(i16 %a) {
; CHECK-LABEL: or16_reg_imm_0xffb3
; CHECK: ori {{r[0-9]+}}, 179
; CHECK: ori {{r[0-9]+}}, 255
    %result = or i16 %a, 65459
    ret i16 %result
}

define i16 @or16_reg_imm_0x00ff(i16 %a) {
; CHECK-LABEL: or16_reg_imm_0x00ff
; CHECK: ori {{r[0-9]+}}, 255
; CHECK-NOT: ori {{r[0-9]+}}, 0
    %result = or i16 %a, 255
    ret i16 %result
}

define i16 @or16_reg_imm_0xb3ff(i16 %a) {
; CHECK-LABEL: or16_reg_imm_0xb3ff
; CHECK: ori {{r[0-9]+}}, 255
; CHECK: ori {{r[0-9]+}}, 179
    %result = or i16 %a, 46079
    ret i16 %result
}
