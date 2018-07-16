; RUN: llc -march=amdgcn -mcpu=gfx900 -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck %s  -check-prefixes=GCN,GFX900
; RUN: llc -march=amdgcn -mcpu=gfx906 -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck %s  -check-prefixes=GCN,GFX906-UNSAFE
; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck %s  -check-prefixes=GCN,GFX906
; RUN: llc -march=amdgcn -mcpu=gfx906 -mattr=-fp64-fp16-denormals,-fp32-denormals -fp-contract=fast -verify-machineinstrs < %s | FileCheck %s  -check-prefixes=GCN,GFX906-CONTRACT
; RUN: llc -march=amdgcn -mcpu=gfx906 -mattr=+fp64-fp16-denormals,+fp32-denormals -fp-contract=fast -verify-machineinstrs < %s | FileCheck %s  -check-prefixes=GCN,GFX906-DENORM-CONTRACT
; (fadd (fmul S1.x, S2.x), (fadd (fmul (S1.y, S2.y), z))) -> (fdot2 S1, S2, z)

; Tests to make sure fdot2 is not generated when vector elements of dot-product expressions
; are not converted from f16 to f32.
; GCN-LABEL: {{^}}dotproduct_f16
; GFX900: v_fma_legacy_f16
; GCN900: v_fma_legacy_f16

; GFX906: v_mul_f16_e32
; GFX906: v_mul_f16_e32

; GFX906-UNSAFE:  v_fma_legacy_f16

; GFX906-CONTRACT: v_mac_f16_e32
; GFX906-DENORM-CONTRACT: v_fma_legacy_f16
define amdgpu_kernel void @dotproduct_f16(<2 x half> addrspace(1)* %src1,
                                          <2 x half> addrspace(1)* %src2,
                                          half addrspace(1)* nocapture %dst) {
entry:
  %src1.vec = load <2 x half>, <2 x half> addrspace(1)* %src1
  %src2.vec = load <2 x half>, <2 x half> addrspace(1)* %src2

  %src1.el1 = extractelement <2 x half> %src1.vec, i64 0
  %src2.el1 = extractelement <2 x half> %src2.vec, i64 0

  %src1.el2 = extractelement <2 x half> %src1.vec, i64 1
  %src2.el2 = extractelement <2 x half> %src2.vec, i64 1

  %mul2 = fmul half %src1.el2, %src2.el2
  %mul1 = fmul half %src1.el1, %src2.el1
  %acc = load half, half addrspace(1)* %dst, align 2
  %acc1 = fadd half %mul2, %acc
  %acc2 = fadd half %mul1, %acc1
  store half %acc2, half addrspace(1)* %dst, align 2
  ret void
}


; We only want to generate fdot2 if vector element of dot product is converted from f16 to f32
; and the vectors are of type <2 x half>
; GCN-LABEL: {{^}}dotproduct_f16_f32
; GFX900: v_mad_mix_f32
; GCN900: v_mad_mix_f32

; GFX906: v_mad_f32
; GFX906: v_mac_f32_e32

; GFX906-UNSAFE: v_dot2_f32_f16

; GFX906-CONTRACT: v_dot2_f32_f16

; GFX906-DENORM-CONTRACT: v_dot2_f32_f16
define amdgpu_kernel void @dotproduct_f16_f32(<2 x half> addrspace(1)* %src1,
                                              <2 x half> addrspace(1)* %src2,
                                              float addrspace(1)* nocapture %dst) {
entry:
  %src1.vec = load <2 x half>, <2 x half> addrspace(1)* %src1
  %src2.vec = load <2 x half>, <2 x half> addrspace(1)* %src2

  %src1.el1 = extractelement <2 x half> %src1.vec, i64 0
  %csrc1.el1 = fpext half %src1.el1 to float
  %src2.el1 = extractelement <2 x half> %src2.vec, i64 0
  %csrc2.el1 = fpext half %src2.el1 to float

  %src1.el2 = extractelement <2 x half> %src1.vec, i64 1
  %csrc1.el2 = fpext half %src1.el2 to float
  %src2.el2 = extractelement <2 x half> %src2.vec, i64 1
  %csrc2.el2 = fpext half %src2.el2 to float

  %mul2 = fmul float %csrc1.el2, %csrc2.el2
  %mul1 = fmul float %csrc1.el1, %csrc2.el1
  %acc = load float, float addrspace(1)* %dst, align 4
  %acc1 = fadd float %mul2, %acc
  %acc2 = fadd float %mul1, %acc1
  store float %acc2, float addrspace(1)* %dst, align 4
  ret void
}

; We only want to generate fdot2 if vector element of dot product is converted from f16 to f32
; and the vectors are of type <2 x half>
; GCN-LABEL: {{^}}dotproduct_diffvecorder
; GFX900: v_mad_mix_f32
; GCN900: v_mad_mix_f32

; GFX906: v_mad_f32
; GFX906: v_mac_f32_e32

; GFX906-UNSAFE: v_dot2_f32_f16

; GFX906-CONTRACT: v_dot2_f32_f16
; GFX906-DENORM-CONTRACT: v_dot2_f32_f16
define amdgpu_kernel void @dotproduct_diffvecorder(<2 x half> addrspace(1)* %src1,
                                                   <2 x half> addrspace(1)* %src2,
                                                   float addrspace(1)* nocapture %dst) {
entry:
  %src1.vec = load <2 x half>, <2 x half> addrspace(1)* %src1
  %src2.vec = load <2 x half>, <2 x half> addrspace(1)* %src2

  %src1.el1 = extractelement <2 x half> %src1.vec, i64 0
  %csrc1.el1 = fpext half %src1.el1 to float
  %src2.el1 = extractelement <2 x half> %src2.vec, i64 0
  %csrc2.el1 = fpext half %src2.el1 to float

  %src1.el2 = extractelement <2 x half> %src1.vec, i64 1
  %csrc1.el2 = fpext half %src1.el2 to float
  %src2.el2 = extractelement <2 x half> %src2.vec, i64 1
  %csrc2.el2 = fpext half %src2.el2 to float

  %mul2 = fmul float %csrc2.el2, %csrc1.el2
  %mul1 = fmul float %csrc1.el1, %csrc2.el1
  %acc = load float, float addrspace(1)* %dst, align 4
  %acc1 = fadd float %mul2, %acc
  %acc2 = fadd float %mul1, %acc1
  store float %acc2, float addrspace(1)* %dst, align 4
  ret void
}

; Tests to make sure dot product is not generated when the vectors are not of <2 x half>.
; GCN-LABEL: {{^}}dotproduct_v4f16
; GFX900: v_mad_mix_f32

; GFX906: v_mad_f32
; GFX906: v_mac_f32_e32

; GFX906-UNSAFE: v_fma_mix_f32

; GFX906-CONTRACT: v_fma_mix_f32
; GFX906-DENORM-CONTRACT: v_fma_mix_f32
define amdgpu_kernel void @dotproduct_v4f16(<4 x half> addrspace(1)* %src1,
                                            <4 x half> addrspace(1)* %src2,
                                            float addrspace(1)* nocapture %dst) {
entry:
  %src1.vec = load <4 x half>, <4 x half> addrspace(1)* %src1
  %src2.vec = load <4 x half>, <4 x half> addrspace(1)* %src2

  %src1.el1 = extractelement <4 x half> %src1.vec, i64 0
  %csrc1.el1 = fpext half %src1.el1 to float
  %src2.el1 = extractelement <4 x half> %src2.vec, i64 0
  %csrc2.el1 = fpext half %src2.el1 to float

  %src1.el2 = extractelement <4 x half> %src1.vec, i64 1
  %csrc1.el2 = fpext half %src1.el2 to float
  %src2.el2 = extractelement <4 x half> %src2.vec, i64 1
  %csrc2.el2 = fpext half %src2.el2 to float

  %mul2 = fmul float %csrc1.el2, %csrc2.el2
  %mul1 = fmul float %csrc1.el1, %csrc2.el1
  %acc = load float, float addrspace(1)* %dst, align 4
  %acc1 = fadd float %mul2, %acc
  %acc2 = fadd float %mul1, %acc1
  store float %acc2, float addrspace(1)* %dst, align 4
  ret void
}

; GCN-LABEL: {{^}}NotAdotproduct
; GFX900: v_mad_mix_f32
; GCN900: v_mad_mix_f32

; GFX906: v_mad_f32
; GFX906: v_mac_f32_e32

; GFX906-UNSAFE: v_fma_mix_f32

; GFX906-CONTRACT: v_fma_mix_f32
; GFX906-DENORM-CONTRACT: v_fma_mix_f32
define amdgpu_kernel void @NotAdotproduct(<2 x half> addrspace(1)* %src1,
                                          <2 x half> addrspace(1)* %src2,
                                          float addrspace(1)* nocapture %dst) {
entry:
  %src1.vec = load <2 x half>, <2 x half> addrspace(1)* %src1
  %src2.vec = load <2 x half>, <2 x half> addrspace(1)* %src2

  %src1.el1 = extractelement <2 x half> %src1.vec, i64 0
  %csrc1.el1 = fpext half %src1.el1 to float
  %src2.el1 = extractelement <2 x half> %src2.vec, i64 0
  %csrc2.el1 = fpext half %src2.el1 to float

  %src1.el2 = extractelement <2 x half> %src1.vec, i64 1
  %csrc1.el2 = fpext half %src1.el2 to float
  %src2.el2 = extractelement <2 x half> %src2.vec, i64 1
  %csrc2.el2 = fpext half %src2.el2 to float

  %mul2 = fmul float %csrc1.el2, %csrc1.el1
  %mul1 = fmul float %csrc2.el1, %csrc2.el2
  %acc = load float, float addrspace(1)* %dst, align 4
  %acc1 = fadd float %mul2, %acc
  %acc2 = fadd float %mul1, %acc1
  store float %acc2, float addrspace(1)* %dst, align 4
  ret void
}

; GCN-LABEL: {{^}}Diff_Idx_NotAdotproduct
; GFX900: v_mad_mix_f32
; GCN900: v_mad_mix_f32

; GFX906: v_mad_f32
; GFX906: v_mac_f32_e32

; GFX906-UNSAFE: v_fma_mix_f32

; GFX906-CONTRACT: v_fma_mix_f32
; GFX906-DENORM-CONTRACT: v_fma_mix_f32
define amdgpu_kernel void @Diff_Idx_NotAdotproduct(<2 x half> addrspace(1)* %src1,
                                                   <2 x half> addrspace(1)* %src2,
                                                   float addrspace(1)* nocapture %dst) {
entry:
  %src1.vec = load <2 x half>, <2 x half> addrspace(1)* %src1
  %src2.vec = load <2 x half>, <2 x half> addrspace(1)* %src2

  %src1.el1 = extractelement <2 x half> %src1.vec, i64 0
  %csrc1.el1 = fpext half %src1.el1 to float
  %src2.el1 = extractelement <2 x half> %src2.vec, i64 0
  %csrc2.el1 = fpext half %src2.el1 to float

  %src1.el2 = extractelement <2 x half> %src1.vec, i64 1
  %csrc1.el2 = fpext half %src1.el2 to float
  %src2.el2 = extractelement <2 x half> %src2.vec, i64 1
  %csrc2.el2 = fpext half %src2.el2 to float

  %mul2 = fmul float %csrc1.el2, %csrc2.el1
  %mul1 = fmul float %csrc1.el1, %csrc2.el2
  %acc = load float, float addrspace(1)* %dst, align 4
  %acc1 = fadd float %mul2, %acc
  %acc2 = fadd float %mul1, %acc1
  store float %acc2, float addrspace(1)* %dst, align 4
  ret void
}