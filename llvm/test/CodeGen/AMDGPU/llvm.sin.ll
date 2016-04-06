; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=SI-SAFE -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -enable-unsafe-fp-math < %s | FileCheck -check-prefix=SI -check-prefix=SI-UNSAFE -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=SI-SAFE -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -enable-unsafe-fp-math < %s | FileCheck -check-prefix=SI -check-prefix=SI-UNSAFE -check-prefix=FUNC %s

; FUNC-LABEL: sin_f32
; EG: MULADD_IEEE *
; EG: FRACT *
; EG: ADD *
; EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG-NOT: SIN
; SI: v_mul_f32
; SI: v_fract_f32
; SI: v_sin_f32
; SI-NOT: v_sin_f32

define void @sin_f32(float addrspace(1)* %out, float %x) #1 {
   %sin = call float @llvm.sin.f32(float %x)
   store float %sin, float addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}sin_3x_f32:
; SI-UNSAFE-NOT: v_add_f32
; SI-UNSAFE: 0x3ef47644
; SI-UNSAFE: v_mul_f32
; SI-SAFE: v_mul_f32
; SI-SAFE: v_mul_f32
; SI: v_fract_f32
; SI: v_sin_f32
; SI-NOT: v_sin_f32
define void @sin_3x_f32(float addrspace(1)* %out, float %x) #1 {
  %y = fmul float 3.0, %x
  %sin = call float @llvm.sin.f32(float %y)
  store float %sin, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sin_2x_f32:
; SI-UNSAFE-NOT: v_add_f32
; SI-UNSAFE: 0x3ea2f983
; SI-UNSAFE: v_mul_f32
; SI-SAFE: v_add_f32
; SI-SAFE: v_mul_f32
; SI: v_fract_f32
; SI: v_sin_f32
; SI-NOT: v_sin_f32
define void @sin_2x_f32(float addrspace(1)* %out, float %x) #1 {
  %y = fmul float 2.0, %x
  %sin = call float @llvm.sin.f32(float %y)
  store float %sin, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_2sin_f32:
; SI-UNSAFE: 0x3ea2f983
; SI-UNSAFE: v_mul_f32
; SI-SAFE: v_add_f32
; SI-SAFE: v_mul_f32
; SI: v_fract_f32
; SI: v_sin_f32
; SI-NOT: v_sin_f32
define void @test_2sin_f32(float addrspace(1)* %out, float %x) #1 {
   %y = fmul float 2.0, %x
   %sin = call float @llvm.sin.f32(float %y)
   store float %sin, float addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}sin_v4f32:
; EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG-NOT: SIN
; SI: v_sin_f32
; SI: v_sin_f32
; SI: v_sin_f32
; SI: v_sin_f32
; SI-NOT: v_sin_f32

define void @sin_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %vx) #1 {
   %sin = call <4 x float> @llvm.sin.v4f32( <4 x float> %vx)
   store <4 x float> %sin, <4 x float> addrspace(1)* %out
   ret void
}

declare float @llvm.sin.f32(float) readnone
declare <4 x float> @llvm.sin.v4f32(<4 x float>) readnone
