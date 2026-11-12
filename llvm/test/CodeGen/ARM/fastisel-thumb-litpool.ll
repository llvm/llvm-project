; RUN: llc -mtriple=thumbv7-apple-ios -O0 -o - %s | FileCheck %s

; We used to accidentally create both an ARM and a Thumb ldr here. It led to an
; assertion failure at the time, but could go all the way through to emission,
; hence the CHECK-NOT.

define i32 @test_thumb_ldrlit() minsize {
; CHECK-LABEL: test_thumb_ldrlit:
; CHECK: ldr r0, LCPI0_0
; CHECK-NOT: ldr
  ret i32 12345678
}
