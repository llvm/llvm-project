; RUN: llc < %s -mtriple=armv6-linux-gnueabi -arm-this-return-forwarding | FileCheck %s -check-prefix=CHECKELF
; RUN: llc < %s -mtriple=thumbv7-apple-ios5.0 -arm-this-return-forwarding | FileCheck %s -check-prefix=CHECKT2D

declare i16 @identity16(i16 returned %x)
declare i32 @identity32(i32 returned %x)
declare zeroext i16 @retzext16(i16 returned %x)
declare i16 @paramzext16(i16 zeroext returned %x)
declare zeroext i16 @bothzext16(i16 zeroext returned %x)

; The zeroext param attribute below is meant to have no effect
define i16 @test_identity(i16 zeroext %x) {
entry:
; CHECKELF-LABEL: test_identity:
; CHECKELF: mov [[SAVEX:r[0-9]+]], r0
; CHECKELF: bl identity16
; CHECKELF: uxth r0, r0
; CHECKELF: bl identity32
; CHECKELF: mov r0, [[SAVEX]]
; CHECKT2D-LABEL: test_identity:
; CHECKT2D: mov [[SAVEX:r[0-9]+]], r0
; CHECKT2D: bl _identity16
; CHECKT2D: uxth r0, r0
; CHECKT2D: bl _identity32
; CHECKT2D: mov r0, [[SAVEX]]
  %call = tail call i16 @identity16(i16 %x)
  %b = zext i16 %call to i32
  %call2 = tail call i32 @identity32(i32 %b)
  ret i16 %x
}

; FIXME: This ought not to require register saving but currently does because
; x is not considered equal to %call (see SelectionDAGBuilder.cpp)
define i16 @test_matched_ret(i16 %x) {
entry:
; CHECKELF-LABEL: test_matched_ret:

; This shouldn't be required
; CHECKELF: mov [[SAVEX:r[0-9]+]], r0

; CHECKELF: bl retzext16
; CHECKELF-NOT: uxth r0, {{r[0-9]+}}
; CHECKELF: bl identity32

; This shouldn't be required
; CHECKELF: mov r0, [[SAVEX]]

; CHECKT2D-LABEL: test_matched_ret:

; This shouldn't be required
; CHECKT2D: mov [[SAVEX:r[0-9]+]], r0

; CHECKT2D: bl _retzext16
; CHECKT2D-NOT: uxth r0, {{r[0-9]+}}
; CHECKT2D: bl _identity32

; This shouldn't be required
; CHECKT2D: mov r0, [[SAVEX]]

  %call = tail call i16 @retzext16(i16 %x)
  %b = zext i16 %call to i32
  %call2 = tail call i32 @identity32(i32 %b)
  ret i16 %x
}

define i16 @test_mismatched_ret(i16 %x) {
entry:
; CHECKELF-LABEL: test_mismatched_ret:
; CHECKELF: mov [[SAVEX:r[0-9]+]], r0
; CHECKELF: bl retzext16
; CHECKELF: sxth r0, {{r[0-9]+}}
; CHECKELF: bl identity32
; CHECKELF: mov r0, [[SAVEX]]
; CHECKT2D-LABEL: test_mismatched_ret:
; CHECKT2D: mov [[SAVEX:r[0-9]+]], r0
; CHECKT2D: bl _retzext16
; CHECKT2D: sxth r0, {{r[0-9]+}}
; CHECKT2D: bl _identity32
; CHECKT2D: mov r0, [[SAVEX]]
  %call = tail call i16 @retzext16(i16 %x)
  %b = sext i16 %call to i32
  %call2 = tail call i32 @identity32(i32 %b)
  ret i16 %x
}

define i16 @test_matched_paramext(i16 %x) {
entry:
; CHECKELF-LABEL: test_matched_paramext:
; CHECKELF: uxth r0, r0
; CHECKELF: bl paramzext16
; CHECKELF: uxth r0, r0
; CHECKELF: bl identity32
; CHECKELF: b paramzext16
; CHECKT2D-LABEL: test_matched_paramext:
; CHECKT2D: uxth r0, r0
; CHECKT2D: bl _paramzext16
; CHECKT2D: uxth r0, r0
; CHECKT2D: bl _identity32
; CHECKT2D: b.w _paramzext16
  %call = tail call i16 @paramzext16(i16 %x)
  %b = zext i16 %call to i32
  %call2 = tail call i32 @identity32(i32 %b)
  %call3 = tail call i16 @paramzext16(i16 %call)
  ret i16 %call3
}

; FIXME: This theoretically ought to optimize to exact same output as the
; version above, but doesn't currently (see SelectionDAGBuilder.cpp) 
define i16 @test_matched_paramext2(i16 %x) {
entry:

; Since there doesn't seem to be an unambiguous optimal selection and
; scheduling of uxth and mov instructions below in lieu of the 'returned'
; optimization, don't bother checking: just verify that the calls are made
; in the correct order as a basic sanity check

; CHECKELF-LABEL: test_matched_paramext2:
; CHECKELF: bl paramzext16
; CHECKELF: bl identity32
; CHECKELF: b paramzext16
; CHECKT2D-LABEL: test_matched_paramext2:
; CHECKT2D: bl _paramzext16
; CHECKT2D: bl _identity32
; CHECKT2D: b.w _paramzext16
  %call = tail call i16 @paramzext16(i16 %x)

; Should make no difference if %x is used below rather than %call, but it does
  %b = zext i16 %x to i32

  %call2 = tail call i32 @identity32(i32 %b)
  %call3 = tail call i16 @paramzext16(i16 %call)
  ret i16 %call3
}

define i16 @test_matched_bothext(i16 %x) {
entry:
; CHECKELF-LABEL: test_matched_bothext:
; CHECKELF: uxth r0, r0
; CHECKELF: bl bothzext16
; CHECKELF-NOT: uxth r0, r0

; FIXME: Tail call should be OK here
; CHECKELF: bl identity32

; CHECKT2D-LABEL: test_matched_bothext:
; CHECKT2D: uxth r0, r0
; CHECKT2D: bl _bothzext16
; CHECKT2D-NOT: uxth r0, r0

; FIXME: Tail call should be OK here
; CHECKT2D: bl _identity32

  %call = tail call i16 @bothzext16(i16 %x)
  %b = zext i16 %x to i32
  %call2 = tail call i32 @identity32(i32 %b)
  ret i16 %call
}

define i16 @test_mismatched_bothext(i16 %x) {
entry:
; CHECKELF-LABEL: test_mismatched_bothext:
; CHECKELF: mov [[SAVEX:r[0-9]+]], r0
; CHECKELF: uxth r0, {{r[0-9]+}}
; CHECKELF: bl bothzext16
; CHECKELF: sxth r0, [[SAVEX]]
; CHECKELF: bl identity32
; CHECKELF: mov r0, [[SAVEX]]
; CHECKT2D-LABEL: test_mismatched_bothext:
; CHECKT2D: mov [[SAVEX:r[0-9]+]], r0
; CHECKT2D: uxth r0, {{r[0-9]+}}
; CHECKT2D: bl _bothzext16
; CHECKT2D: sxth r0, [[SAVEX]]
; CHECKT2D: bl _identity32
; CHECKT2D: mov r0, [[SAVEX]]
  %call = tail call i16 @bothzext16(i16 %x)
  %b = sext i16 %x to i32
  %call2 = tail call i32 @identity32(i32 %b)
  ret i16 %x
}
