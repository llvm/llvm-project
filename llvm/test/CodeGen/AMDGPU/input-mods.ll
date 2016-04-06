;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG
;RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=CM

;EG-LABEL: {{^}}test:
;EG: EXP_IEEE *
;CM-LABEL: {{^}}test:
;CM: EXP_IEEE T{{[0-9]+}}.X, -|T{{[0-9]+}}.X|
;CM: EXP_IEEE T{{[0-9]+}}.Y (MASKED), -|T{{[0-9]+}}.X|
;CM: EXP_IEEE T{{[0-9]+}}.Z (MASKED), -|T{{[0-9]+}}.X|
;CM: EXP_IEEE * T{{[0-9]+}}.W (MASKED), -|T{{[0-9]+}}.X|

define amdgpu_ps void @test(<4 x float> inreg %reg0) {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = call float @llvm.fabs.f32(float %r0)
   %r2 = fsub float -0.000000e+00, %r1
   %r3 = call float @llvm.exp2.f32(float %r2)
   %vec = insertelement <4 x float> undef, float %r3, i32 0
   call void @llvm.R600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

declare float @llvm.exp2.f32(float) readnone
declare float @llvm.fabs.f32(float) readnone
declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
