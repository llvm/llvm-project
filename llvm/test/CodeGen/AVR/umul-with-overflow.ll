; RUN: llc -mattr=avr6 < %s -march=avr | FileCheck %s

define i1 @unsigned_multiplication_did_overflow(i8, i8) unnamed_addr {
; CHECK-LABEL: unsigned_multiplication_did_overflow:
entry-block:
  %2 = tail call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %0, i8 %1)
  %3 = extractvalue { i8, i1 } %2, 1
  ret i1 %3

; Multiply, return if the high byte is zero
;
; CHECK: mul    r{{[0-9]+}}, r{{[0-9]+}}
; CHECK: mov    [[HIGH:r[0-9]+]], r1
; CHECK: ldi    [[RET:r[0-9]+]], 1
; CHECK: cpi    {{.*}}[[HIGH]], 0
; CHECK: brne   [[LABEL:.LBB[_0-9]+]]
; CHECK: mov    {{.*}}[[RET]], r1
; CHECK: {{.*}}[[LABEL]]
; CHECK: ret
}

declare { i8, i1 } @llvm.umul.with.overflow.i8(i8, i8)
