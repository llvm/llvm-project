; RUN: llc -mtriple=linux-arm-gnueabihf -mattr=+neon %s -o - | FileCheck %s

; Check that no intermediate integer register is used.
define i32 @no-intermediate-register-for-zero-imm(double %x) #0 {
entry:
; CHECK-LABEL: no-intermediate-register-for-zero-imm
; CHECK-NOT: vmov
; CHECK: vcmp
  %cmp = fcmp une double %x, 0.000000e+00
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
