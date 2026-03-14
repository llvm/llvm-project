; RUN: llc -mtriple arm-gnueabi -mattr=+v6t2,+hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,HWDIV
; RUN: llc -mtriple arm-gnueabi -mattr=+v6t2,-hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,SOFT-AEABI
; RUN: llc -mtriple arm-gnu -mattr=+v6t2,+hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,HWDIV
; RUN: llc -mtriple arm-gnu -mattr=+v6t2,-hwdiv-arm -global-isel %s -o - | FileCheck %s -check-prefixes=CHECK,SOFT-DEFAULT

define arm_aapcscc i32 @test_sdiv_i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_sdiv_i32:
; HWDIV: sdiv
; SOFT-AEABI: bl __aeabi_idiv
; SOFT-DEFAULT: bl __divsi3
  %r = sdiv i32 %a, %b
  ret i32 %r
}

define arm_aapcscc i32 @test_udiv_i32(i32 %a, i32 %b) {
; CHECK-LABEL: test_udiv_i32:
; HWDIV: udiv
; SOFT-AEABI: bl __aeabi_uidiv
; SOFT-DEFAULT: bl __udivsi3
  %r = udiv i32 %a, %b
  ret i32 %r
}

define arm_aapcscc i16 @test_sdiv_i16(i16 %a, i16 %b) {
; CHECK-LABEL: test_sdiv_i16:
; HWDIV: sdiv
; SOFT-AEABI: bl __aeabi_idiv
; SOFT-DEFAULT: bl __divsi3
  %r = sdiv i16 %a, %b
  ret i16 %r
}

define arm_aapcscc i16 @test_udiv_i16(i16 %a, i16 %b) {
; CHECK-LABEL: test_udiv_i16:
; HWDIV: udiv
; SOFT-AEABI: bl __aeabi_uidiv
; SOFT-DEFAULT: bl __udivsi3
  %r = udiv i16 %a, %b
  ret i16 %r
}

define arm_aapcscc i8 @test_sdiv_i8(i8 %a, i8 %b) {
; CHECK-LABEL: test_sdiv_i8:
; HWDIV: sdiv
; SOFT-AEABI: bl __aeabi_idiv
; SOFT-DEFAULT: bl __divsi3
  %r = sdiv i8 %a, %b
  ret i8 %r
}

define arm_aapcscc i8 @test_udiv_i8(i8 %a, i8 %b) {
; CHECK-LABEL: test_udiv_i8:
; HWDIV: udiv
; SOFT-AEABI: bl __aeabi_uidiv
; SOFT-DEFAULT: bl __udivsi3
  %r = udiv i8 %a, %b
  ret i8 %r
}

define arm_aapcscc i32 @test_srem_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_srem_i32:
; HWDIV: sdiv
; SOFT-AEABI: bl __aeabi_idivmod
; SOFT-DEFAULT: bl __modsi3
  %r = srem i32 %x, %y
  ret i32 %r
}

define arm_aapcscc i32 @test_urem_i32(i32 %x, i32 %y) {
; CHECK-LABEL: test_urem_i32:
; HWDIV: udiv
; SOFT-AEABI: bl __aeabi_uidivmod
; SOFT-DEFAULT: bl __umodsi3
  %r = urem i32 %x, %y
  ret i32 %r
}

define arm_aapcscc i16 @test_srem_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_srem_i16:
; HWDIV: sdiv
; SOFT-AEABI: bl __aeabi_idivmod
; SOFT-DEFAULT: bl __modsi3
  %r = srem i16 %x, %y
  ret i16 %r
}

define arm_aapcscc i16 @test_urem_i16(i16 %x, i16 %y) {
; CHECK-LABEL: test_urem_i16:
; HWDIV: udiv
; SOFT-AEABI: bl __aeabi_uidivmod
; SOFT-DEFAULT: bl __umodsi3
  %r = urem i16 %x, %y
  ret i16 %r
}

define arm_aapcscc i8 @test_srem_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_srem_i8:
; HWDIV: sdiv
; SOFT-AEABI: bl __aeabi_idivmod
; SOFT-DEFAULT: bl __modsi3
  %r = srem i8 %x, %y
  ret i8 %r
}

define arm_aapcscc i8 @test_urem_i8(i8 %x, i8 %y) {
; CHECK-LABEL: test_urem_i8:
; HWDIV: udiv
; SOFT-AEABI: bl __aeabi_uidivmod
; SOFT-DEFAULT: bl __umodsi3
  %r = urem i8 %x, %y
  ret i8 %r
}
