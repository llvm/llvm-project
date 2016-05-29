; RUN: llc < %s -mtriple=i686-unknown | FileCheck %s

; These tests just check that the plumbing is in place for @llvm.bitreverse. The
; actual output is massive at the moment as llvm.bitreverse is not yet legal.

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>) readnone

define <2 x i16> @f(<2 x i16> %a) {
; CHECK-LABEL: f:
; CHECK: shll
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  ret <2 x i16> %b
}

declare i8 @llvm.bitreverse.i8(i8) readnone

define i8 @g(i8 %a) {
; CHECK-LABEL: g:
; CHECK: shlb
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  ret i8 %b
}

; These tests check that bitreverse(constant) calls are folded

define <2 x i16> @fold_v2i16() {
; CHECK-LABEL: fold_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    movw $-4096, %ax # imm = 0xF000
; CHECK-NEXT:    movw $240, %dx
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> <i16 15, i16 3840>)
  ret <2 x i16> %b
}

define i8 @fold_i8() {
; CHECK-LABEL: fold_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb $-16, %al
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 15)
  ret i8 %b
}

; These tests check that bitreverse(bitreverse()) calls are removed

define i8 @identity_i8(i8 %a) {
; CHECK-LABEL: identity_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movb {{[0-9]+}}(%esp), %al
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 %a)
  %c = call i8 @llvm.bitreverse.i8(i8 %b)
  ret i8 %c
}

define <2 x i16> @identity_v2i16(<2 x i16> %a) {
; CHECK-LABEL: identity_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzwl {{[0-9]+}}(%esp), %eax
; CHECK-NEXT:    movzwl {{[0-9]+}}(%esp), %edx
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %a)
  %c = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %b)
  ret <2 x i16> %c
}

; These tests check that bitreverse(undef) calls are removed

define i8 @undef_i8() {
; CHECK-LABEL: undef_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    retl
  %b = call i8 @llvm.bitreverse.i8(i8 undef)
  ret i8 %b
}

define <2 x i16> @undef_v2i16() {
; CHECK-LABEL: undef_v2i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    retl
  %b = call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> undef)
  ret <2 x i16> %b
}
