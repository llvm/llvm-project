; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: {{^}}fma_f32:
; SI: v_fma_f32 {{v[0-9]+, v[0-9]+, v[0-9]+, v[0-9]+}}

; EG: MEM_RAT_{{.*}} STORE_{{.*}} [[RES:T[0-9]\.[XYZW]]], {{T[0-9]\.[XYZW]}},
; EG: FMA {{\*? *}}[[RES]]
define void @fma_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                     float addrspace(1)* %in2, float addrspace(1)* %in3) {
  %r0 = load float, float addrspace(1)* %in1
  %r1 = load float, float addrspace(1)* %in2
  %r2 = load float, float addrspace(1)* %in3
  %r3 = tail call float @llvm.fma.f32(float %r0, float %r1, float %r2)
  store float %r3, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fma_v2f32:
; SI: v_fma_f32
; SI: v_fma_f32

; EG: MEM_RAT_{{.*}} STORE_{{.*}} [[RES:T[0-9]]].[[CHLO:[XYZW]]][[CHHI:[XYZW]]], {{T[0-9]\.[XYZW]}},
; EG-DAG: FMA {{\*? *}}[[RES]].[[CHLO]]
; EG-DAG: FMA {{\*? *}}[[RES]].[[CHHI]]
define void @fma_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in1,
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

; EG: MEM_RAT_{{.*}} STORE_{{.*}} [[RES:T[0-9]]].{{[XYZW][XYZW][XYZW][XYZW]}}, {{T[0-9]\.[XYZW]}},
; EG-DAG: FMA {{\*? *}}[[RES]].X
; EG-DAG: FMA {{\*? *}}[[RES]].Y
; EG-DAG: FMA {{\*? *}}[[RES]].Z
; EG-DAG: FMA {{\*? *}}[[RES]].W
define void @fma_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in1,
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
define void @fma_commute_mul_inline_imm_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) nounwind {
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
define void @fma_commute_mul_s_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b, float %b) nounwind {
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
