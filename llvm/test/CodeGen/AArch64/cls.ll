; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s

; @llvm.aarch64.cls must be directly translated into the 'cls' instruction

; CHECK-LABEL: cls
; CHECK: cls [[REG:w[0-9]+]], [[REG]]
define i32 @cls(i32 %t) {
  %cls.i = call i32 @llvm.aarch64.cls(i32 %t)
  ret i32 %cls.i
}

; CHECK-LABEL: cls64
; CHECK: cls [[REG:x[0-9]+]], [[REG]]
define i32 @cls64(i64 %t) {
  %cls.i = call i32 @llvm.aarch64.cls64(i64 %t)
  ret i32 %cls.i
}

declare i32 @llvm.aarch64.cls(i32) nounwind
declare i32 @llvm.aarch64.cls64(i64) nounwind

define i8 @cls_i8(i8 %x) {
; CHECK-LABEL: cls_i8:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sxtb w8, w0 
; CHECK-NEXT:    cls w8, w8 
; CHECK-NEXT:    sub w0, w8, #24 
; CHECK-NEXT:    ret

  %a = ashr i8 %x, 7
  %b = xor i8 %x, %a
  %c = call i8 @llvm.ctlz.i8(i8 %b, i1 false)
  %d = sub i8 %c, 1
  ret i8 %d
}

; The result is in the range [1-31], so we don't need an andi after the cls.
define i32 @cls_i32_knownbits(i32 %x) {
; CHECK-LABEL: cls_i32_knownbits:
; CHECK:       // %bb.0:
; CHECK-NEXT:    cls	w0, w0
; CHECK-NEXT:    ret
  %a = ashr i32 %x, 31
  %b = xor i32 %x, %a
  %c = call i32 @llvm.ctlz.i32(i32 %b, i1 false)
  %d = sub i32 %c, 1
  %e = and i32 %d, 31
  ret i32 %e
}

; There are at least 16 redundant sign bits so we don't need an ori after the clsw.
define i32 @cls_i32_knownbits_2(i16 signext %x) {
; CHECK-LABEL: cls_i32_knownbits_2:
; CHECK:       // %bb.0:
; CHECK-NEXT:    cls w0, w0
; CHECK-NEXT:    ret
  %sext = sext i16 %x to i32
  %a = ashr i32 %sext, 31
  %b = xor i32 %sext, %a
  %c = call i32 @llvm.ctlz.i32(i32 %b, i1 false)
  %d = sub i32 %c, 1
  %e = or i32 %d, 16
  ret i32 %e
}

define i32 @cls_i32_knownbits_3(i8 signext %x) {
; CHECK-LABEL: cls_i32_knownbits_3:
; CHECK:       // %bb.0:
; CHECK-NEXT:    cls	w0, w0
; CHECK-NEXT:    ret
  %sext = sext i8 %x to i32
  %a = ashr i32 %sext, 31
  %b = xor i32 %sext, %a
  %c = call i32 @llvm.ctlz.i32(i32 %b, i1 false)
  %d = sub i32 %c, 1
  %e = or i32 %d, 24
  ret i32 %e
}

; Negative test. We only know there is at least 1 redundant sign bit. We can't
; remove the ori.
define i32 @cls_i32_knownbits_4(i32 signext %x) {
; CHECK-LABEL: cls_i32_knownbits_4:
; CHECK:       // %bb.0:
; CHECK-NEXT:   sbfx	w8, w0, #0, #31
; CHECK-NEXT:	  cls	w8, w8
; CHECK-NEXT:	  orr	w0, w8, #0x1
; CHECK-NEXT:	  ret
  %shl = shl i32 %x, 1
  %ashr = ashr i32 %shl, 1
  %a = ashr i32 %ashr, 31
  %b = xor i32 %ashr, %a
  %c = call i32 @llvm.ctlz.i32(i32 %b, i1 false)
  %d = sub i32 %c, 1
  %e = or i32 %d, 1
  ret i32 %e
 }
