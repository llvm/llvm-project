;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_ps void @test(<4 x float> inreg %reg0) {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = fdiv float 1.0, %r0
   %vec = insertelement <4 x float> undef, float %r1, i32 0
   call void @llvm.R600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)
