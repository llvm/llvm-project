;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: MAX * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_ps void @test(<4 x float> inreg %reg0) {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = fcmp oge float %r0, %r1
   %r3 = select i1 %r2, float %r0, float %r1
   %vec = insertelement <4 x float> undef, float %r3, i32 0
   call void @llvm.R600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
