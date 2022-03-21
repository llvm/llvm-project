; RUN: llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx906 -stop-after=amdgpu-isel < %s | FileCheck -check-prefix=GFX906 %s

; GCN-LABEL: name:            uniform_vec_0_i16
; GCN: S_LSHL_B32
define amdgpu_kernel void @uniform_vec_0_i16(i32 addrspace(1)* %out, i16 %a) {
  %tmp = insertelement <2 x i16> undef, i16 0, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %a, i32 1
  %val = bitcast <2 x i16> %vec to i32
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_vec_0_i16
; GCN: V_LSHLREV_B32_e64
define i32 @divergent_vec_0_i16(i16 %a) {
  %tmp = insertelement <2 x i16> undef, i16 0, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %a, i32 1
  %val = bitcast <2 x i16> %vec to i32
  ret i32 %val
}

; GCN-LABEL: name:            uniform_vec_i16_0
; GCN: S_AND_B32
define amdgpu_kernel void @uniform_vec_i16_0(i32 addrspace(1)* %out, i16 %a) {
  %tmp = insertelement <2 x i16> undef, i16 %a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 0, i32 1
  %val = bitcast <2 x i16> %vec to i32
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_vec_i16_0
; GCN: V_AND_B32_e64
define i32 @divergent_vec_i16_0(i16 %a) {
  %tmp = insertelement <2 x i16> undef, i16 %a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 0, i32 1
  %val = bitcast <2 x i16> %vec to i32
  ret i32 %val
}

; GCN-LABEL: name:            uniform_vec_f16_0
; GCN: S_AND_B32
define amdgpu_kernel void @uniform_vec_f16_0(float addrspace(1)* %out, half %a) {
  %tmp = insertelement <2 x half> undef, half %a, i32 0
  %vec = insertelement <2 x half> %tmp, half 0.0, i32 1
  %val = bitcast <2 x half> %vec to float
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_vec_f16_0
; GCN: V_CVT_F16_F32_e64 0, %0
; GCN: COPY %1

; GFX9-LABEL: name:            divergent_vec_f16_0
; GFX9: V_AND_B32_e64
define float @divergent_vec_f16_0(half %a) {
  %tmp = insertelement <2 x half> undef, half %a, i32 0
  %vec = insertelement <2 x half> %tmp, half 0.0, i32 1
  %val = bitcast <2 x half> %vec to float
  ret float %val
}

; GCN-LABEL: name:            uniform_vec_i16_LL
; GCN: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 65535
; GCN: %[[AND:[0-9]+]]:sreg_32 = S_AND_B32 killed %{{[0-9]+}}, killed %[[IMM]]
; GCN: %[[SHIFT:[0-9]+]]:sreg_32 = S_MOV_B32 16
; GCN:  %[[SHL:[0-9]+]]:sreg_32 = S_LSHL_B32 killed %{{[0-9]+}}, killed %[[SHIFT]]
; GCN: S_OR_B32 killed %[[AND]], killed %[[SHL]]

; GFX9-LABEL: name:            uniform_vec_i16_LL
; GFX9: S_PACK_LL_B32_B16
define amdgpu_kernel void @uniform_vec_i16_LL(i32 addrspace(4)* %in0, i32 addrspace(4)* %in1) {
  %val0 = load volatile i32, i32 addrspace(4)* %in0
  %val1 = load volatile i32, i32 addrspace(4)* %in1
  %lo = trunc i32 %val0 to i16
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: name:            divergent_vec_i16_LL
; GCN: %[[SHIFT:[0-9]+]]:sreg_32 = S_MOV_B32 16
; GCN: %[[SHL:[0-9]+]]:vgpr_32 = V_LSHLREV_B32_e64 killed %[[SHIFT]], %1, implicit $exec
; GCN: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 65535
; GCN: %[[AND:[0-9]+]]:vgpr_32 = V_AND_B32_e64 %0, killed %[[IMM]], implicit $exec
; GCN: V_OR_B32_e64 killed %[[AND]], killed %[[SHL]], implicit $exec

; GFX9-LABEL: name:            divergent_vec_i16_LL
; GFX9: %[[IMM:[0-9]+]]:vgpr_32 = V_MOV_B32_e32 65535
; GFX9: %[[AND:[0-9]+]]:vgpr_32 = V_AND_B32_e64 killed %[[IMM]]
; GFX9: V_LSHL_OR_B32_e64 %{{[0-9]+}}, 16, killed %[[AND]]
define i32 @divergent_vec_i16_LL(i16 %a, i16 %b) {
  %tmp = insertelement <2 x i16> undef, i16 %a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %b, i32 1
  %val = bitcast <2 x i16> %vec to i32
  ret i32 %val
}

; GCN-LABEL: name:            uniform_vec_i16_LH
; GCN-DAG: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 65535
; GCN-DAG: %[[AND:[0-9]+]]:sreg_32 = S_AND_B32 killed %{{[0-9]+}}, killed %[[IMM]]
; GCN-DAG: %[[NEG:[0-9]+]]:sreg_32 = S_MOV_B32 -65536
; GCN-DAG: %[[ANDN:[0-9]+]]:sreg_32 = S_AND_B32 killed %{{[0-9]+}}, killed %[[NEG]]
; GCN: S_OR_B32 killed %[[AND]], killed %[[ANDN]]

; GFX9-LABEL: name:            uniform_vec_i16_LH
; GFX9: S_PACK_LH_B32_B16
define amdgpu_kernel void @uniform_vec_i16_LH(i32 addrspace(1)* %out, i16 %a, i32 %b) {
  %shift = lshr i32 %b, 16
  %tr = trunc i32 %shift to i16
  %tmp = insertelement <2 x i16> undef, i16 %a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %tr, i32 1
  %val = bitcast <2 x i16> %vec to i32
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_vec_i16_LH
; GCN: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 65535
; GCN: V_BFI_B32_e64 killed %[[IMM]]
define i32 @divergent_vec_i16_LH(i16 %a, i32 %b) {
  %shift = lshr i32 %b, 16
  %tr = trunc i32 %shift to i16
  %tmp = insertelement <2 x i16> undef, i16 %a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %tr, i32 1
  %val = bitcast <2 x i16> %vec to i32
  ret i32 %val
}

; GCN-LABEL: name:            uniform_vec_i16_HH
; GCN: %[[SHIFT:[0-9]+]]:sreg_32 = S_MOV_B32 16
; GCN:  %[[SHR:[0-9]+]]:sreg_32 = S_LSHR_B32 killed %{{[0-9]+}}, killed %[[SHIFT]]
; GCN: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 -65536
; GCN: %[[AND:[0-9]+]]:sreg_32 = S_AND_B32 killed %{{[0-9]+}}, killed %[[IMM]]
; GCN: S_OR_B32 killed %[[SHR]], killed %[[AND]]

; GFX9-LABEL: name:            uniform_vec_i16_HH
; GFX9: S_PACK_HH_B32_B16
define amdgpu_kernel void @uniform_vec_i16_HH(i32 addrspace(1)* %out, i32 %a, i32 %b) {
  %shift_a = lshr i32 %a, 16
  %tr_a = trunc i32 %shift_a to i16
  %shift_b = lshr i32 %b, 16
  %tr_b = trunc i32 %shift_b to i16
  %tmp = insertelement <2 x i16> undef, i16 %tr_a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %tr_b, i32 1
  %val = bitcast <2 x i16> %vec to i32
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: name:            divergent_vec_i16_HH
; GCN: %[[SHR:[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 killed %{{[0-9]+}}, %0, implicit $exec
; GCN: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 -65536
; GCN: %[[AND:[0-9]+]]:vgpr_32 = V_AND_B32_e64 %1, killed %[[IMM]], implicit $exec
; GCN: V_OR_B32_e64 killed %[[SHR]], killed %[[AND]], implicit $exec

; GFX9-LABEL: name:            divergent_vec_i16_HH
; GFX9: %[[SHR:[0-9]+]]:vgpr_32 = V_LSHRREV_B32_e64 16, %0
; GFX9: %[[IMM:[0-9]+]]:vgpr_32 = V_MOV_B32_e32 -65536, implicit $exec
; GFX9: V_AND_OR_B32_e64 %1, killed %[[IMM]], killed %[[SHR]]
define i32 @divergent_vec_i16_HH(i32 %a, i32 %b) {
  %shift_a = lshr i32 %a, 16
  %tr_a = trunc i32 %shift_a to i16
  %shift_b = lshr i32 %b, 16
  %tr_b = trunc i32 %shift_b to i16
  %tmp = insertelement <2 x i16> undef, i16 %tr_a, i32 0
  %vec = insertelement <2 x i16> %tmp, i16 %tr_b, i32 1
  %val = bitcast <2 x i16> %vec to i32
  ret i32 %val
}

; GCN-LABEL: name:            uniform_vec_f16_LL
; GCN: %[[IMM:[0-9]+]]:sreg_32 = S_MOV_B32 65535
; GCN: %[[AND:[0-9]+]]:sreg_32 = S_AND_B32 killed %{{[0-9]+}}, killed %[[IMM]]
; GCN: %[[SHIFT:[0-9]+]]:sreg_32 = S_MOV_B32 16
; GCN:  %[[SHL:[0-9]+]]:sreg_32 = S_LSHL_B32 killed %{{[0-9]+}}, killed %[[SHIFT]]
; GCN: S_OR_B32 killed %[[AND]], killed %[[SHL]]

; GFX9-LABEL: name:            uniform_vec_f16_LL
; GFX9: S_PACK_LL_B32_B16
define amdgpu_kernel void @uniform_vec_f16_LL(i32 addrspace(4)* %in0, i32 addrspace(4)* %in1) {
  %val0 = load volatile i32, i32 addrspace(4)* %in0
  %val1 = load volatile i32, i32 addrspace(4)* %in1
  %lo.i = trunc i32 %val0 to i16
  %hi.i = trunc i32 %val1 to i16
  %lo = bitcast i16 %lo.i to half
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: name:            divergent_vec_f16_LL
; GCN: %[[SHIFT:[0-9]+]]:sreg_32 = S_MOV_B32 16
; GCN: %[[SHL:[0-9]+]]:vgpr_32 = V_LSHLREV_B32_e64 killed %[[SHIFT]]
; GCN: V_OR_B32_e64 killed %{{[0-9]+}}, killed %[[SHL]], implicit $exec

; GFX9-LABEL: name:            divergent_vec_f16_LL
; GFX9: %[[IMM:[0-9]+]]:vgpr_32 = V_MOV_B32_e32 65535
; GFX9: %[[AND:[0-9]+]]:vgpr_32 = V_AND_B32_e64 killed %[[IMM]]
; GFX9: V_LSHL_OR_B32_e64 %{{[0-9]+}}, 16, killed %[[AND]]
define float @divergent_vec_f16_LL(half %a, half %b) {
  %tmp = insertelement <2 x half> undef, half %a, i32 0
  %vec = insertelement <2 x half> %tmp, half %b, i32 1
  %val = bitcast <2 x half> %vec to float
  ret float %val
}

; GFX906-LABEL: name:            build_vec_v2i16_undeflo_divergent
; GFX906: %[[LOAD:[0-9]+]]:vgpr_32 = DS_READ_U16
; GFX906: %{{[0-9]+}}:vgpr_32 = COPY %[[LOAD]]
define <2 x i16> @build_vec_v2i16_undeflo_divergent(i16 addrspace(3)* %in) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build = insertelement <2 x i16> undef, i16 %load, i32 0
  ret <2 x i16> %build
}

; GFX906-LABEL: name:            build_vec_v2i16_undeflo_uniform
; GFX906: %[[LOAD:[0-9]+]]:vgpr_32 = DS_READ_U16
; GFX906: %{{[0-9]+}}:sreg_32 = COPY %[[LOAD]]
define amdgpu_kernel void @build_vec_v2i16_undeflo_uniform(i16 addrspace(3)* %in, i32 addrspace(1)* %out) #0 {
entry:
  %load = load i16, i16 addrspace(3)* %in
  %build = insertelement <2 x i16> undef, i16 %load, i32 0
  %result = bitcast <2 x i16> %build to i32
  store i32 %result, i32 addrspace(1)* %out
  ret void
}
