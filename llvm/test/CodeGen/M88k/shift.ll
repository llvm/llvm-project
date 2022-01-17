; Test shift instructions.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 | FileCheck %s

; Check two register operands.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: mak %r2, %r2, %r3
; CHECK: jmp %r1
  %res = shl i32 %a, %b
  ret i32 %res
}

; Check two register operands.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: extu %r2, %r2, %r3
; CHECK: jmp %r1
  %res = lshr i32 %a, %b
  ret i32 %res
}

; Check two register operands.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: ext %r2, %r2, %r3
; CHECK: jmp %r1
  %res = ashr i32 %a, %b
  ret i32 %res
}

; Check immediate operand.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: mak %r2, %r2, 0<2>
; CHECK: jmp %r1
  %res = shl i32 %a, 2
  ret i32 %res
}

; Check combine logical shift right and and.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: extu %r2, %r2, 5<1>
; CHECK: jmp %r1
  %shift = lshr i32 %a, 1
  %res = and i32 %shift, 31
  ret i32 %res
}

; Check combine arithmetic shift right and and.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: extu %r2, %r2, 5<1>
; CHECK: jmp %r1
  %shift = ashr i32 %a, 1
  %res = and i32 %shift, 31
  ret i32 %res
}

; Check combine and and logical shift right.
define i32 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: extu %r2, %r2, 5<1>
; CHECK: jmp %r1
  %and = and i32 %a, 62
  %res = lshr i32 %and, 1
  ret i32 %res
}

; Check combine and and arithmetic shift right.
define i32 @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: extu %r2, %r2, 5<1>
; CHECK: jmp %r1
  %and = and i32 %a, 62
  %res = ashr i32 %and, 1
  ret i32 %res
}

; Check combine and and shift left.
define i32 @f9(i32 %a) {
; CHECK-LABEL: f9:
; CHECK: mak %r2, %r2, 8<15>
; CHECK: jmp %r1
  %and = and i32 %a, 255
  %res = shl i32 %and, 15
  ret i32 %res
}

; Check combine shift left and logical shift right.
define i32 @f10(i32 %a) {
; CHECK-LABEL: f10:
; CHECK: extu %r2, %r2, 16<8>
; CHECK: jmp %r1
  %shl = shl i32 %a, 8
  %res = lshr i32 %shl, 16
  ret i32 %res
}

; Check combine shift left and arithmetic shift right.
define i32 @f11(i32 %a) {
; CHECK-LABEL: f11:
; CHECK: ext %r2, %r2, 16<8>
; CHECK: jmp %r1
  %shl = shl i32 %a, 8
  %res = ashr i32 %shl, 16
  ret i32 %res
}