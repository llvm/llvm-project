; RUN: llc -show-mc-encoding < %s | FileCheck %s

; Test that the direct object emission selects the and variant with 8 bit
; immediate.
; We used to get this wrong when using direct object emission, but not when
; reading assembly.


target triple = "x86_64-pc-linux"

define void @f1() {
; CHECK-LABEL: f1:
; CHECK: andq    $-32, %rsp              # encoding: [0x48,0x83,0xe4,0xe0]
  %foo = alloca i8, align 32
  ret void
}

define void @f2(i16 %x, i1 *%y) {
; CHECK-LABEL: f2:
; CHECK: andl	$1, %edi                # encoding: [0x83,0xe7,0x01]
  %c = trunc i16 %x to i1
  store i1 %c, i1* %y
  ret void
}

define void @f3(i32 %x, i1 *%y) {
; CHECK-LABEL: f3:
; CHECK: andl	$1, %edi                # encoding: [0x83,0xe7,0x01]
  %c = trunc i32 %x to i1
  store i1 %c, i1* %y
  ret void
}
