; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -ppc-asm-full-reg-names < %s | FileCheck -check-prefix=CHECK-FN %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -ppc-reg-with-percent-prefix < %s | FileCheck -check-prefix=CHECK-PN %s

define i64 @test1(i64 %a, i64 %b) {
; CHECK-LABEL: @test1
; CHECK-FN-LABEL: @test1

entry:
  ret i64 %b

; CHECK: mr 3, 4
; CHECK-FN: mr r3, r4
; CHECK-PN: mr %r3, %r4

; CHECK: blr
; CHECK-FN: blr
; CHECK-PN: blr
}

