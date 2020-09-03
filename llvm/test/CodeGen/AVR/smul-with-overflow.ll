; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

define i1 @signed_multiplication_did_overflow(i8, i8) unnamed_addr {
; CHECK-LABEL: signed_multiplication_did_overflow:
entry-block:
  %2 = tail call { i8, i1 } @llvm.smul.with.overflow.i8(i8 %0, i8 %1)
  %3 = extractvalue { i8, i1 } %2, 1
  ret i1 %3

; Multiply, fill the low byte with the sign of the low byte via
; arithmetic shifting, compare it to the high byte.
;
; CHECK: muls   r24, r22
; CHECK: mov    [[HIGH:r[0-9]+]], r1
; CHECK: mov    [[LOW:r[0-9]+]], r0
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: asr    {{.*}}[[LOW]]
; CHECK: ldi    [[RET:r[0-9]+]], 1
; CHECK: cp     {{.*}}[[HIGH]], {{.*}}[[LOW]]
; CHECK: brne   [[LABEL:.LBB[_0-9]+]]
; CHECK: ldi    {{.*}}[[RET]], 0
; CHECK: {{.*}}[[LABEL]]
; CHECK: ret
}

declare { i8, i1 } @llvm.smul.with.overflow.i8(i8, i8)
