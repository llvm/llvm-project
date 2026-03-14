; The hwdiv subtarget feature should only influence thumb, not arm.
; RUN: llc < %s -mtriple=arm-gnueabi -mattr=+hwdiv | FileCheck %s -check-prefixes=ALL,AEABI-NOHWDIV
; RUN: llc < %s -mtriple=arm-gnueabi -mattr=-hwdiv | FileCheck %s -check-prefixes=ALL,AEABI-NOHWDIV
; RUN: llc < %s -mtriple=thumbv7-gnueabi -mattr=+hwdiv | FileCheck %s -check-prefixes=ALL,THUMB-HWDIV
; RUN: llc < %s -mtriple=thumbv7-gnueabi -mattr=-hwdiv | FileCheck %s -check-prefixes=ALL,AEABI-NOHWDIV

; The hwdiv-arm subtarget feature should only influence arm, not thumb.
; RUN: llc < %s -mtriple=arm-gnueabi -mattr=+hwdiv-arm | FileCheck %s -check-prefixes=ALL,ARM-HWDIV
; RUN: llc < %s -mtriple=arm-gnueabi -mattr=-hwdiv-arm | FileCheck %s -check-prefixes=ALL,AEABI-NOHWDIV
; RUN: llc < %s -mtriple=thumbv7-gnueabi -mattr=+hwdiv-arm | FileCheck %s -check-prefixes=ALL,AEABI-NOHWDIV
; RUN: llc < %s -mtriple=thumbv7-gnueabi -mattr=-hwdiv-arm | FileCheck %s -check-prefixes=ALL,AEABI-NOHWDIV

define arm_aapcscc i32 @test_i32_srem(i32 %x, i32 %y) {
; ALL-LABEL: test_i32_srem:
; ARM-HWDIV: sdiv [[Q:r[0-9]+]], r0, r1
; ARM-HWDIV: mul [[P:r[0-9]+]], [[Q]], r1
; ARM-HWDIV: sub r0, r0, [[P]]
; THUMB-HWDIV: sdiv [[Q:r[0-9]+]], r0, r1
; THUMB-HWDIV: mls r0, [[Q]], r1, r0
; AEABI-NOHWDIV: bl __aeabi_idivmod
; AEABI-NOHWDIV: mov r0, r1
  %r = srem i32 %x, %y
  ret i32 %r
}

define arm_aapcscc i32 @test_i32_urem(i32 %x, i32 %y) {
; ALL-LABEL: test_i32_urem:
; ARM-HWDIV: udiv [[Q:r[0-9]+]], r0, r1
; ARM-HWDIV: mul [[P:r[0-9]+]], [[Q]], r1
; ARM-HWDIV: sub r0, r0, [[P]]
; THUMB-HWDIV: udiv [[Q:r[0-9]+]], r0, r1
; THUMB-HWDIV: mls r0, [[Q]], r1, r0
; AEABI-NOHWDIV: bl __aeabi_uidivmod
; AEABI-NOHWDIV: mov r0, r1
  %r = urem i32 %x, %y
  ret i32 %r
}
