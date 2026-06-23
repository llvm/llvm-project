; RUN: llc -asm-verbose=false --mtriple=armv7-none-eabihf --mattr=+vfp3 < %s | FileCheck %s --check-prefix=CHECK-VFP
; RUN: llc -asm-verbose=false --mtriple=armv7-none-eabi --mattr=-fpregs < %s | FileCheck %s -check-prefix=CHECK-NOVFP
; RUN: llc -asm-verbose=false --mtriple=armv7-none-eabihf --mattr=-fpregs < %s 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR -check-prefix=CHECK-NOVFP

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"

; CHECK-VFP: vadd.f32
; CHECK-ERROR: The hard-float ABI can't be used for a target that doesn't support floating-point (ignoring float-abi)
; CHECK-NOVFP: bl __aeabi_fadd
define float @test_fadd(float %a, float %b) #0 {
  %r = fadd float %a, %b
  ret float %r
}

attributes #0 = { nounwind }
