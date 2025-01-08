;RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s

;CHECK-LABEL: test1:
;CHECK: LOG_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}},
;CHECK-NEXT: MUL NON-IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PS}},
;CHECK-NEXT: EXP_IEEE * T{{[0-9]+\.[XYZW], PV\.[XYZW]}},

define amdgpu_ps void @test1(<4 x float> inreg %reg0) {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = call float @llvm.pow.f32( float %r0, float %r1)
   %vec = insertelement <4 x float> undef, float %r2, i32 0
   call void @llvm.r600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

;CHECK-LABEL: test2:
;CHECK: LOG_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}},
;CHECK-NEXT: MUL NON-IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PS}},
;CHECK-NEXT: LOG_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}},
;CHECK-NEXT: MUL NON-IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PS}},
;CHECK-NEXT: EXP_IEEE * T{{[0-9]+\.[XYZW], PV\.[XYZW]}},
;CHECK-NEXT: EXP_IEEE * T{{[0-9]+\.[XYZW], PV\.[XYZW]}},
;CHECK-NEXT: LOG_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}},
;CHECK-NEXT: MUL NON-IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PS}},
;CHECK-NEXT: LOG_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}},
;CHECK-NEXT: MUL NON-IEEE T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], PS}},
;CHECK-NEXT: EXP_IEEE * T{{[0-9]+\.[XYZW], PV\.[XYZW]}},
;CHECK-NEXT: EXP_IEEE * T{{[0-9]+\.[XYZW], PV\.[XYZW]}},
define amdgpu_ps void @test2(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
   %vec = call <4 x float> @llvm.pow.v4f32( <4 x float> %reg0, <4 x float> %reg1)
   call void @llvm.r600.store.swizzle(<4 x float> %vec, i32 0, i32 0)
   ret void
}

declare float @llvm.pow.f32(float ,float ) readonly
declare <4 x float> @llvm.pow.v4f32(<4 x float> ,<4 x float> ) readonly
declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)
