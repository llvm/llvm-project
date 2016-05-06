; RUN: llc < %s -march=amdgcn | FileCheck %s -check-prefix=SI -check-prefix=FUNC
; RUN: llc < %s -march=amdgcn -mcpu=tonga | FileCheck %s -check-prefix=SI -check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s -check-prefix=EG -check-prefix=FUNC

;FUNC-LABEL: test
;EG: MULADD_IEEE *
;EG: FRACT *
;EG: ADD *
;EG: COS * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG-NOT: COS
;SI: v_cos_f32
;SI-NOT: v_cos_f32

define void @test(float addrspace(1)* %out, float %x) #1 {
   %cos = call float @llvm.cos.f32(float %x)
   store float %cos, float addrspace(1)* %out
   ret void
}

;FUNC-LABEL: testv
;EG: COS * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG: COS * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG: COS * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG: COS * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
;EG-NOT: COS
;SI: v_cos_f32
;SI: v_cos_f32
;SI: v_cos_f32
;SI: v_cos_f32
;SI-NOT: v_cos_f32

define void @testv(<4 x float> addrspace(1)* %out, <4 x float> inreg %vx) #1 {
   %cos = call <4 x float> @llvm.cos.v4f32(<4 x float> %vx)
   store <4 x float> %cos, <4 x float> addrspace(1)* %out
   ret void
}

declare float @llvm.cos.f32(float) readnone
declare <4 x float> @llvm.cos.v4f32(<4 x float>) readnone
