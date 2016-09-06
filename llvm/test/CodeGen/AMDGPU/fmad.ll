;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: MULADD_IEEE * {{T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_ps void @test(<4 x float> inreg %reg0) {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = extractelement <4 x float> %reg0, i32 2
   %r3 = fmul float %r0, %r1
   %r4 = fadd float %r3, %r2
   %vec = insertelement <4 x float> undef, float %r4, i32 0
   call void @llvm.r600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

declare float @fabs(float ) readnone
declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)
