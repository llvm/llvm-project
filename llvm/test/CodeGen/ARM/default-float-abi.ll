; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s --check-prefix=CHECK-HARD
; RUN: llc -mtriple=armv7-linux-musleabihf %s -o - | FileCheck %s --check-prefix=CHECK-HARD
; RUN: llc -mtriple=armv7-linux-eabihf %s -o - | FileCheck %s --check-prefix=CHECK-HARD
; RUN: llc -mtriple=armv7-linux-gnueabihf -float-abi=soft %s -o - | FileCheck %s --check-prefix=CHECK-SOFT
; RUN: llc -mtriple=armv7-linux-musleabihf -float-abi=soft %s -o - | FileCheck %s --check-prefix=CHECK-SOFT
; RUN: llc -mtriple=armv7-linux-gnueabi %s -o - | FileCheck %s --check-prefix=CHECK-SOFT
; RUN: llc -mtriple=armv7-linux-musleabi %s -o - | FileCheck %s --check-prefix=CHECK-SOFT
; RUN: llc -mtriple=armv7-linux-eabi -float-abi=hard %s -o - | FileCheck %s --check-prefix=CHECK-HARD
; RUN: llc -mtriple=thumbv7-apple-ios6.0 %s -o - | FileCheck %s --check-prefix=CHECK-SOFT

define float @test_abi(float %lhs, float %rhs) {
  %sum = fadd float %lhs, %rhs
  ret float %sum

; CHECK-HARD-LABEL: test_abi:
; CHECK-HARD-NOT: vmov
; CHECK-HARD: vadd.f32 s0, s0, s1
; CHECK-HARD-NOT: vmov

; CHECK-SOFT-LABEL: test_abi:
; CHECK-SOFT-DAG: vmov [[LHS:s[0-9]+]], r0
; CHECK-SOFT-DAG: vmov [[RHS:s[0-9]+]], r1
; CHECK-SOFT: vadd.f32 [[DEST:s[0-9]+]], [[LHS]], [[RHS]]
; CHECK-SOFT: vmov r0, [[DEST]]
}
