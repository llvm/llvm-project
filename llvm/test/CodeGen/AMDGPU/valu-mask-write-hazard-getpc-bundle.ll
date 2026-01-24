; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 < %s | FileCheck %s
;
; This test validates that the VALU mask write hazard pass correctly handles
; bundled instructions.

; CHECK-LABEL: main:
; CHECK: s_getpc_b64
; CHECK: s_waitcnt_depctr

define amdgpu_cs i32 @main(i32 %arg, i32 %arg1) {
bb:
  %i = udiv i32 %arg, %arg1
  %i2 = uitofp i32 %i to float
  %i3 = udiv i32 1, %arg
  %i4 = uitofp i32 %i3 to float
  %i5 = call float @llvm.fma.f32(float %i2, float 0.000000e+00, float %i4)
  %i6 = fptosi float %i5 to i32
  %i7 = call i32 @func(i32 0, i32 %i6)
  ret i32 %i7
}

declare i32 @func(i32, i32)
