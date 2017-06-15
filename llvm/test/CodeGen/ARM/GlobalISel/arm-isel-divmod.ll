; We use V6 ops so we can easily check for the extensions (sxth vs bit tricks).
; RUN: llc -mtriple arm-gnueabi -mattr=+v6,+hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,HWDIV
; RUN: llc -mtriple arm-gnueabi -mattr=+v6,-hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,SOFT-AEABI
; RUN: llc -mtriple arm-gnu -mattr=+v6,+hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,HWDIV
; RUN: llc -mtriple arm-gnu -mattr=+v6,-hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,SOFT-DEFAULT

define arm_aapcscc i32 @test_sdiv_i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_sdiv_i32:
; HWDIV: sdiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_idiv
; SOFT-DEFAULT: blx __divsi3
  %r = sdiv i32 %a, %b
  ret i32 %r
}

define arm_aapcscc i32 @test_udiv_i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_udiv_i32:
; HWDIV: udiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_uidiv
; SOFT-DEFAULT: blx __udivsi3
  %r = udiv i32 %a, %b
  ret i32 %r
}

define arm_aapcscc i16 @test_sdiv_i16(i16 %a, i16 %b) {
; CHECK-LABEL: test_sdiv_i16:
; CHECK-DAG: sxth r0, r0
; CHECK-DAG: sxth r1, r1
; HWDIV: sdiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_idiv
; SOFT-DEFAULT: blx __divsi3
  %r = sdiv i16 %a, %b
  ret i16 %r
}

define arm_aapcscc i16 @test_udiv_i16(i16 %a, i16 %b) {
; CHECK-LABEL: test_udiv_i16:
; CHECK-DAG: uxth r0, r0
; CHECK-DAG: uxth r1, r1
; HWDIV: udiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_uidiv
; SOFT-DEFAULT: blx __udivsi3
  %r = udiv i16 %a, %b
  ret i16 %r
}

define arm_aapcscc i8 @test_sdiv_i8(i8 %a, i8 %b) {
; CHECK-LABEL: test_sdiv_i8:
; CHECK-DAG: sxtb r0, r0
; CHECK-DAG: sxtb r1, r1
; HWDIV: sdiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_idiv
; SOFT-DEFAULT: blx __divsi3
  %r = sdiv i8 %a, %b
  ret i8 %r
}

define arm_aapcscc i8 @test_udiv_i8(i8 %a, i8 %b) {
; CHECK-LABEL: test_udiv_i8:
; CHECK-DAG: uxtb r0, r0
; CHECK-DAG: uxtb r1, r1
; HWDIV: udiv r0, r0, r1
; SOFT-AEABI: blx __aeabi_uidiv
; SOFT-DEFAULT: blx __udivsi3
  %r = udiv i8 %a, %b
  ret i8 %r
}

define arm_aapcscc i32 @test_srem_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_srem_i32:
; HWDIV: sdiv [[Q:r[0-9]+]], r0, r1
; HWDIV: mul [[P:r[0-9]+]], [[Q]], r1
; HWDIV: sub r0, r0, [[P]]
; SOFT-AEABI: blx __aeabi_idivmod
; SOFT-DEFAULT: blx __modsi3
  %r = srem i32 %x, %y
  ret i32 %r
}

define arm_aapcscc i32 @test_urem_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_urem_i32:
; HWDIV: udiv [[Q:r[0-9]+]], r0, r1
; HWDIV: mul [[P:r[0-9]+]], [[Q]], r1
; HWDIV: sub r0, r0, [[P]]
; SOFT-AEABI: blx __aeabi_uidivmod
; SOFT-DEFAULT: blx __umodsi3
  %r = urem i32 %x, %y
  ret i32 %r
}
