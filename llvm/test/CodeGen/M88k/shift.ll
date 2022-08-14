; Test shift instructions.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -verify-machineinstrs | FileCheck --check-prefixes=CHECK,MC88100 %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -verify-machineinstrs | FileCheck --check-prefixes=CHECK,MC88110 %s

; Check two register operands.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: jmp.n %r1
; CHECK: mak %r2, %r2, %r3
  %res = shl i32 %a, %b
  ret i32 %res
}

; Check two register operands.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: jmp.n %r1
; CHECK: extu %r2, %r2, %r3
  %res = lshr i32 %a, %b
  ret i32 %res
}

; Check two register operands.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: jmp.n %r1
; CHECK: ext %r2, %r2, %r3
  %res = ashr i32 %a, %b
  ret i32 %res
}

; Check immediate operand.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: jmp.n %r1
; CHECK: mak %r2, %r2, 0<2>
  %res = shl i32 %a, 2
  ret i32 %res
}

; Check combine logical shift right and and.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: jmp.n %r1
; CHECK: extu %r2, %r2, 5<1>
  %shift = lshr i32 %a, 1
  %res = and i32 %shift, 31
  ret i32 %res
}

; Check combine arithmetic shift right and and.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: jmp.n %r1
; CHECK: extu %r2, %r2, 5<1>
  %shift = ashr i32 %a, 1
  %res = and i32 %shift, 31
  ret i32 %res
}

; Check combine and and logical shift right.
define i32 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: jmp.n %r1
; CHECK: extu %r2, %r2, 5<1>
  %and = and i32 %a, 62
  %res = lshr i32 %and, 1
  ret i32 %res
}

; Check combine and and arithmetic shift right.
define i32 @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: jmp.n %r1
; CHECK: extu %r2, %r2, 5<1>
  %and = and i32 %a, 62
  %res = ashr i32 %and, 1
  ret i32 %res
}

; Check combine and and shift left.
define i32 @f9(i32 %a) {
; CHECK-LABEL: f9:
; CHECK: jmp.n %r1
; CHECK: mak %r2, %r2, 8<15>
  %and = and i32 %a, 255
  %res = shl i32 %and, 15
  ret i32 %res
}

; Check combine shift left and logical shift right.
define i32 @f10(i32 %a) {
; CHECK-LABEL: f10:
; CHECK: jmp.n %r1
; CHECK: extu %r2, %r2, 16<8>
  %shl = shl i32 %a, 8
  %res = lshr i32 %shl, 16
  ret i32 %res
}

; Check combine shift left and arithmetic shift right.
define i32 @f11(i32 %a) {
; CHECK-LABEL: f11:
; CHECK: jmp.n %r1
; CHECK: ext %r2, %r2, 16<8>
  %shl = shl i32 %a, 8
  %res = ashr i32 %shl, 16
  ret i32 %res
}

; Check rotate right with constant.
define i32 @f12(i32 %a) {
; CHECK-LABEL: f12:
; CHECK: jmp.n %r1
; CHECK: rot %r2, %r2, <3>
  %res = call i32 @llvm.fshr.i32(i32 %a, i32 %a, i32 3)
  ret i32 %res
}

; Check rotate right with register operand.
define i32 @f13(i32 %a, i32 %b) {
; CHECK-LABEL: f13:
; CHECK: jmp.n %r1
; CHECK: rot %r2, %r2, %r3
  %res = call i32 @llvm.fshr.i32(i32 %a, i32 %a, i32 %b)
  ret i32 %res
}

; Check rotate left with constant.
define i32 @f14(i32 %a) {
; CHECK-LABEL: f14:
; CHECK: jmp.n %r1
; CHECK: rot %r2, %r2, <3>
  %res = call i32 @llvm.fshl.i32(i32 %a, i32 %a, i32 29)
  ret i32 %res
}

define i32 @f15(i32 %a, i32 %b) {
; CHECK-LABEL: f15:
; CHECK: jmp.n %r1
; CHECK: lda.h %r2, %r2[%r3]
  %shl = shl i32 %b, 1
  %res = add i32 %a, %shl
  ret i32 %res
}

define i32 @f16(i32 %a, i32 %b) {
; CHECK-LABEL: f16:
; CHECK: jmp.n %r1
; CHECK: lda %r2, %r2[%r3]
  %shl = shl i32 %b, 2
  %res = add i32 %a, %shl
  ret i32 %res
}

define i32 @f17(i32 %a, i32 %b) {
; CHECK-LABEL: f17:
; MC88100: mak %r3, %r3, 0<3>
; CHECK:   jmp.n %r1
; MC88100: addu %r2, %r2, %r3
; MC88110: lda.x %r2, %r2[%r3]
  %shl = shl i32 %b, 3
  %res = add i32 %a, %shl
  ret i32 %res
}

declare i32 @llvm.fshr.i32(i32, i32, i32)
declare i32 @llvm.fshl.i32(i32, i32, i32)
