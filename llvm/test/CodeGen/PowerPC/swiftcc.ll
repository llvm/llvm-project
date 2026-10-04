; RUN: llc -mtriple=powerpc-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; swiftcc is lowered like the C convention on 32-bit PowerPC. Check that it
; is accepted at all: LowerCall_32SVR4 used to assert on it.

define swiftcc i32 @swiftcc_param(i32 %a, i32 %b) {
; CHECK-LABEL: swiftcc_param:
; CHECK: blr
  %r = add i32 %a, %b
  ret i32 %r
}

define swiftcc i32 @call_swiftcc(i32 %a, i32 %b) {
; CHECK-LABEL: call_swiftcc:
; CHECK: bl swiftcc_param
  %r = call swiftcc i32 @swiftcc_param(i32 %a, i32 %b)
  ret i32 %r
}

define swiftcc ptr @swiftself_param(ptr swiftself %addr) {
; CHECK-LABEL: swiftself_param:
; CHECK: blr
  ret ptr %addr
}
