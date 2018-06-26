; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX906 -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=cedar -verify-machineinstrs < %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=juniper -verify-machineinstrs < %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood -verify-machineinstrs < %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=sumo -verify-machineinstrs < %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=barts -verify-machineinstrs < %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=caicos -verify-machineinstrs < %s
; RUN:  not llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=turks -verify-machineinstrs < %s

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: {{^}}fma_f32:
; SI: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
; GFX906: v_fmac_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; EG: MEM_RAT_{{.*}} STORE_{{.*}} [[RES:T[0-9]\.[XYZW]]], {{T[0-9]\.[XYZW]}},
; EG: FMA {{\*? *}}[[RES]]
define amdgpu_kernel void @fma_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                     float addrspace(1)* %in2, float addrspace(1)* %in3) {
  %r0 = load float, float addrspace(1)* %in1
  %r1 = load float, float addrspace(1)* %in2
  %r2 = load float, float addrspace(1)* %in3
  %r3 = tail call float @llvm.fma.f32(float %r0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fmac_to_3addr_f32:
; GCN: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
define float @fmac_to_3addr_f32(float %r0, float %r1, float %r2) {
  %r3 = tail call float @llvm.fma.f32(float %r0, float %r1, float %r2)
  ret float %r3
}

; FUNC-LABEL: {{^}}fma_v2f32:
; SI: v_fma_f32
; SI: v_fma_f32

; GFX906: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
; GFX906: v_fmac_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+}}

; EG: MEM_RAT_{{.*}} STORE_{{.*}} [[RES:T[0-9]]].[[CHLO:[XYZW]]][[CHHI:[XYZW]]], {{T[0-9]\.[XYZW]}},
; EG-DAG: FMA {{\*? *}}[[RES]].[[CHLO]]
; EG-DAG: FMA {{\*? *}}[[RES]].[[CHHI]]
define amdgpu_kernel void @fma_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in1,
                       <2 x float> addrspace(1)* %in2, <2 x float> addrspace(1)* %in3) {
  %r0 = load <2 x float>, <2 x float> addrspace(1)* %in1
  %r1 = load <2 x float>, <2 x float> addrspace(1)* %in2
  %r2 = load <2 x float>, <2 x float> addrspace(1)* %in3
  %r3 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %r0, <2 x float> %r1, <2 x float> %r2)
  store <2 x float> %r3, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fma_v4f32:
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_fma_f32
; GFX906: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
; GFX906: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
; GFX906: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}
; GFX906: v_fmac_f32_e32 {{v[0-9]+, v[0-9]+, v[0-9]+$}}

; EG: MEM_RAT_{{.*}} STORE_{{.*}} [[RES:T[0-9]]].{{[XYZW][XYZW][XYZW][XYZW]}}, {{T[0-9]\.[XYZW]}},
; EG-DAG: FMA {{\*? *}}[[RES]].X
; EG-DAG: FMA {{\*? *}}[[RES]].Y
; EG-DAG: FMA {{\*? *}}[[RES]].Z
; EG-DAG: FMA {{\*? *}}[[RES]].W
define amdgpu_kernel void @fma_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in1,
                       <4 x float> addrspace(1)* %in2, <4 x float> addrspace(1)* %in3) {
  %r0 = load <4 x float>, <4 x float> addrspace(1)* %in1
  %r1 = load <4 x float>, <4 x float> addrspace(1)* %in2
  %r2 = load <4 x float>, <4 x float> addrspace(1)* %in3
  %r3 = tail call <4 x float> @llvm.fma.v4f32(<4 x float> %r0, <4 x float> %r1, <4 x float> %r2)
  store <4 x float> %r3, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @fma_commute_mul_inline_imm_f32
; SI: v_fma_f32 {{v[0-9]+}}, {{v[0-9]+}}, 2.0, {{v[0-9]+}}
define amdgpu_kernel void @fma_commute_mul_inline_imm_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) nounwind {
  %tid = tail call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %fma = call float @llvm.fma.f32(float %a, float 2.0, float %b)
  store float %fma, float addrspace(1)* %out.gep, align 4
  ret void
}

; FUNC-LABEL: @fma_commute_mul_s_f32
define amdgpu_kernel void @fma_commute_mul_s_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b, float %b) nounwind {
  %tid = tail call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %c = load float, float addrspace(1)* %in.b.gep, align 4

  %fma = call float @llvm.fma.f32(float %a, float %b, float %c)
  store float %fma, float addrspace(1)* %out.gep, align 4
  ret void
}

; Without special casing the inline constant check for v_fmac_f32's
; src2, this fails to fold the 1.0 into an fma.

; FUNC-LABEL: {{^}}fold_inline_imm_into_fmac_src2_f32:
; GFX906: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX906: {{buffer|flat|global}}_load_dword [[B:v[0-9]+]]

; GFX906: v_add_f32_e32 [[TMP2:v[0-9]+]], [[A]], [[A]]
; GFX906: v_fma_f32 v{{[0-9]+}}, [[TMP2]], -4.0, 1.0
define amdgpu_kernel void @fold_inline_imm_into_fmac_src2_f32(float addrspace(1)* %out, float addrspace(1)* %a, float addrspace(1)* %b) nounwind {
bb:
  %tid = call i32 @llvm.r600.read.tidig.x()
  %tid.ext = sext i32 %tid to i64
  %gep.a = getelementptr inbounds float, float addrspace(1)* %a, i64 %tid.ext
  %gep.b = getelementptr inbounds float, float addrspace(1)* %b, i64 %tid.ext
  %gep.out = getelementptr inbounds float, float addrspace(1)* %out, i64 %tid.ext
  %tmp = load volatile float, float addrspace(1)* %gep.a
  %tmp1 = load volatile float, float addrspace(1)* %gep.b
  %tmp2 = fadd contract float %tmp, %tmp
  %tmp3 = fmul contract float %tmp2, 4.0
  %tmp4 = fsub contract float 1.0, %tmp3
  %tmp5 = fadd contract float %tmp4, %tmp1
  %tmp6 = fadd contract float %tmp1, %tmp1
  %tmp7 = fmul contract float %tmp6, %tmp
  %tmp8 = fsub contract float 1.0, %tmp7
  %tmp9 = fmul contract float %tmp8, 8.0
  %tmp10 = fadd contract float %tmp5, %tmp9
  store float %tmp10, float addrspace(1)* %gep.out
  ret void
}
