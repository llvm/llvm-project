; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

; 2x64 vector should be returned in Q0.

define <2 x double> @test(ptr %p) nounwind {
; CHECK: test
; CHECK: ldr q0, [x0]
; CHECK: ret
  %tmp1 = load <2 x double>, ptr %p, align 16
  ret <2 x double> %tmp1
}
