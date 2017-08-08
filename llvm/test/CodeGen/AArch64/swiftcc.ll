; RUN: llc -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=aarch64-apple-ios -o - %s | FileCheck %s

; CHECK: t1
; CHECK: fadd s0, s0, s1
; CHECK: ret
define swiftcc float @t1(float %a, float %b) {
entry:
  %add = fadd float %a, %b
  ret float %add
}
