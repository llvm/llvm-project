; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s

; FUNC-LABEL: sin_f32
; EG: MULADD_IEEE *
; EG: FRACT *
; EG: ADD *
; EG: SIN * T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG-NOT: SIN

; GCN: v_mul_f32
; SICIVI: v_fract_f32
; GFX9-NOT: v_fract_f32
; GCN: v_sin_f32
; GCN-NOT: v_sin_f32
define amdgpu_kernel void @sin_f32(float addrspace(1)* %out, float %x) #1 {
   %sin = call float @llvm.sin.f32(float %x)
   store float %sin, float addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}safe_sin_3x_f32:
; GCN: v_mul_f32
; GCN: v_mul_f32
; SICIVI: v_fract_f32
; GFX9-NOT: v_fract_f32
; GCN: v_sin_f32
; GCN-NOT: v_sin_f32
define amdgpu_kernel void @safe_sin_3x_f32(float addrspace(1)* %out, float %x) #1 {
  %y = fmul float 3.0, %x
  %sin = call float @llvm.sin.f32(float %y)
  store float %sin, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}unsafe_sin_3x_f32:
; GCN-NOT: v_add_f32
; GCN: 0x3ef47644
; GCN: v_mul_f32
; SICIVI: v_fract_f32
; GFX9-NOT: v_fract_f32
; GCN: v_sin_f32
; GCN-NOT: v_sin_f32
define amdgpu_kernel void @unsafe_sin_3x_f32(float addrspace(1)* %out, float %x) #2 {
  %y = fmul float 3.0, %x
  %sin = call float @llvm.sin.f32(float %y)
  store float %sin, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}safe_sin_2x_f32:
; GCN: v_add_f32
; GCN: v_mul_f32
; SICIVI: v_fract_f32
; GFX9-NOT: v_fract_f32
; GCN: v_sin_f32
; GCN-NOT: v_sin_f32
define amdgpu_kernel void @safe_sin_2x_f32(float addrspace(1)* %out, float %x) #1 {
  %y = fmul float 2.0, %x
  %sin = call float @llvm.sin.f32(float %y)
  store float %sin, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}unsafe_sin_2x_f32:
; GCN-NOT: v_add_f32
; GCN: 0x3ea2f983
; GCN: v_mul_f32
; SICIVI: v_fract_f32
; GFX9-NOT: v_fract_f32
; GCN: v_sin_f32
; GCN-NOT: v_sin_f32
define amdgpu_kernel void @unsafe_sin_2x_f32(float addrspace(1)* %out, float %x) #2 {
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

; GCN: v_sin_f32
; GCN: v_sin_f32
; GCN: v_sin_f32
; GCN: v_sin_f32
; GCN-NOT: v_sin_f32
define amdgpu_kernel void @sin_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %vx) #1 {
   %sin = call <4 x float> @llvm.sin.v4f32( <4 x float> %vx)
   store <4 x float> %sin, <4 x float> addrspace(1)* %out
   ret void
}

declare float @llvm.sin.f32(float) #0
declare <4 x float> @llvm.sin.v4f32(<4 x float>) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "unsafe-fp-math"="false" }
attributes #2 = { nounwind "unsafe-fp-math"="true" }
