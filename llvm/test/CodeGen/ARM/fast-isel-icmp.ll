; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

define i32 @icmp_i16_unsigned(i16 %a, i16 %b) nounwind {
entry:
; ARM: icmp_i16_unsigned
; ARM: uxth r0, r0
; ARM: uxth r1, r1
; ARM: cmp	r0, r1
; THUMB: icmp_i16_unsigned
; THUMB: uxth r0, r0
; THUMB: uxth r1, r1
; THUMB: cmp	r0, r1
  %cmp = icmp ult i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i8_signed(i8 %a, i8 %b) nounwind {
entry:
; ARM: icmp_i8_signed
; ARM: sxtb r0, r0
; ARM: sxtb r1, r1
; ARM: cmp r0, r1
; THUMB: icmp_i8_signed
; THUMB: sxtb r0, r0
; THUMB: sxtb r1, r1
; THUMB: cmp r0, r1
  %cmp = icmp sgt i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define i32 @icmp_i1_unsigned(i1 %a, i1 %b) nounwind {
entry:
; ARM: icmp_i1_unsigned
; ARM: and r0, r0, #1
; ARM: and r1, r1, #1
; ARM: cmp r0, r1
; THUMB: icmp_i1_unsigned
; THUMB: and r0, r0, #1
; THUMB: and r1, r1, #1
; THUMB: cmp r0, r1
  %cmp = icmp ult i1 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}
