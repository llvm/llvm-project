; RUN: llc < %s -mtriple=avr | FileCheck %s

; Unit test for: PR 31345

define i16 @and16_reg_imm_0xff00(i16 %a) {
; CHECK-LABEL: and16_reg_imm_0xff00
; CHECK: andi {{r[0-9]+}}, 0
; CHECK-NOT: andi {{r[0-9]+}}, 255
    %result = and i16 %a, 65280
    ret i16 %result
}

define i16 @and16_reg_imm_0xffb3(i16 %a) {
; CHECK-LABEL: and16_reg_imm_0xffb3
; CHECK: andi {{r[0-9]+}}, 179
; CHECK-NOT: andi {{r[0-9]+}}, 255
    %result = and i16 %a, 65459
    ret i16 %result
}

define i16 @and16_reg_imm_0x00ff(i16 %a) {
; CHECK-LABEL: and16_reg_imm_0x00ff
; CHECK-NOT: andi {{r[0-9]+}}, 255
; CHECK: andi {{r[0-9]+}}, 0
    %result = and i16 %a, 255
    ret i16 %result
}

define i16 @and16_reg_imm_0xb3ff(i16 %a) {
; CHECK-LABEL: and16_reg_imm_0xb3ff
; CHECK-NOT: andi {{r[0-9]+}}, 255
; CHECK: andi {{r[0-9]+}}, 179
    %result = and i16 %a, 46079
    ret i16 %result
}

define i16 @and16_reg_imm_0xffff(i16 %a) {
; CHECK-LABEL: and16_reg_imm_0xffff
; CHECK-NOT: andi {{r[0-9]+}}, 255
; CHECK-NOT: andi {{r[0-9]+}}, 255
    %result = and i16 %a, 65535
    ret i16 %result
}

define i16 @and16_reg_imm_0xabcd(i16 %a) {
; CHECK-LABEL: and16_reg_imm_0xabcd
; CHECK: andi {{r[0-9]+}}, 205
; CHECK: andi {{r[0-9]+}}, 171
    %result = and i16 %a, 43981
    ret i16 %result
}
