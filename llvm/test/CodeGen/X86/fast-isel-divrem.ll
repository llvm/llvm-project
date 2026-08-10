; RUN: llc -mtriple=x86_64-none-linux -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=i686-none-linux -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define i8 @test_sdiv8(i8 %dividend, i8 %divisor) nounwind {
entry:
  %result = sdiv i8 %dividend, %divisor
  ret i8 %result
}

; CHECK-LABEL: test_sdiv8:
; CHECK: movsbl
; CHECK: idivb

define i8 @test_srem8(i8 %dividend, i8 %divisor) nounwind {
entry:
  %result = srem i8 %dividend, %divisor
  ret i8 %result
}

; CHECK-LABEL: test_srem8:
; CHECK: movsbl
; CHECK: idivb

define i8 @test_udiv8(i8 %dividend, i8 %divisor) nounwind {
entry:
  %result = udiv i8 %dividend, %divisor
  ret i8 %result
}

; CHECK-LABEL: test_udiv8:
; CHECK: movzbl
; CHECK: divb

define i8 @test_urem8(i8 %dividend, i8 %divisor) nounwind {
entry:
  %result = urem i8 %dividend, %divisor
  ret i8 %result
}

; CHECK-LABEL: test_urem8:
; CHECK: movzbl
; CHECK: divb

define i16 @test_sdiv16(i16 %dividend, i16 %divisor) nounwind {
entry:
  %result = sdiv i16 %dividend, %divisor
  ret i16 %result
}

; CHECK-LABEL: test_sdiv16:
; CHECK: cwtd
; CHECK: idivw

define i16 @test_srem16(i16 %dividend, i16 %divisor) nounwind {
entry:
  %result = srem i16 %dividend, %divisor
  ret i16 %result
}

; CHECK-LABEL: test_srem16:
; CHECK: cwtd
; CHECK: idivw

define i16 @test_udiv16(i16 %dividend, i16 %divisor) nounwind {
entry:
  %result = udiv i16 %dividend, %divisor
  ret i16 %result
}

; CHECK-LABEL: test_udiv16:
; CHECK: xorl
; CHECK: divw

define i16 @test_urem16(i16 %dividend, i16 %divisor) nounwind {
entry:
  %result = urem i16 %dividend, %divisor
  ret i16 %result
}

; CHECK-LABEL: test_urem16:
; CHECK: xorl
; CHECK: divw

define i32 @test_sdiv32(i32 %dividend, i32 %divisor) nounwind {
entry:
  %result = sdiv i32 %dividend, %divisor
  ret i32 %result
}

; CHECK-LABEL: test_sdiv32:
; CHECK: cltd
; CHECK: idivl

define i32 @test_srem32(i32 %dividend, i32 %divisor) nounwind {
entry:
  %result = srem i32 %dividend, %divisor
  ret i32 %result
}

; CHECK-LABEL: test_srem32:
; CHECK: cltd
; CHECK: idivl

define i32 @test_udiv32(i32 %dividend, i32 %divisor) nounwind {
entry:
  %result = udiv i32 %dividend, %divisor
  ret i32 %result
}

; CHECK-LABEL: test_udiv32:
; CHECK: xorl
; CHECK: divl

define i32 @test_urem32(i32 %dividend, i32 %divisor) nounwind {
entry:
  %result = urem i32 %dividend, %divisor
  ret i32 %result
}

; CHECK-LABEL: test_urem32:
; CHECK: xorl
; CHECK: divl
