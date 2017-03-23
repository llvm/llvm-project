; RUN: llc -march=amdgcn -mattr=-fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-fp64-fp16-denormals,-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}mac_f16:
; GCN: {{buffer|flat}}_load_ushort v[[A_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort v[[B_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort v[[C_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32:[0-9]+]], v[[C_F16]]
; SI:  v_mac_f32_e32 v[[C_F32]], v[[B_F32]], v[[A_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[C_F32]]
; SI:  buffer_store_short v[[R_F16]]
; VI:  v_mac_f16_e32 v[[C_F16]], v[[B_F16]], v[[A_F16]]
; VI:  buffer_store_short v[[C_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_same_add:
; SI:  v_mad_f32 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD:v[0-9]+]]
; SI:  v_mac_f32_e32 [[ADD]], v{{[0-9]+}}, v{{[0-9]+}}

; VI:  v_mad_f16 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD:v[0-9]+]]
; VI:  v_mac_f16_e32 [[ADD]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_same_add(
    half addrspace(1)* %r0,
    half addrspace(1)* %r1,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c,
    half addrspace(1)* %d,
    half addrspace(1)* %e) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c
  %d.val = load half, half addrspace(1)* %d
  %e.val = load half, half addrspace(1)* %e

  %t0.val = fmul half %a.val, %b.val
  %r0.val = fadd half %t0.val, %c.val

  %t1.val = fmul half %d.val, %e.val
  %r1.val = fadd half %t1.val, %c.val

  store half %r0.val, half addrspace(1)* %r0
  store half %r1.val, half addrspace(1)* %r1
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_a:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_mad_f32 v{{[0-9]+}}, -[[CVT_A]], [[CVT_B]], [[CVT_C]]

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_a(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %a.neg = fsub half -0.0, %a.val
  %t.val = fmul half %a.neg, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_b:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_mad_f32 v{{[0-9]+}}, -[[CVT_A]], [[CVT_B]], [[CVT_C]]

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_b(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %b.neg = fsub half -0.0, %b.val
  %t.val = fmul half %a.val, %b.neg
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_c:
; SI: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; SI: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_c(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %c.neg = fsub half -0.0, %c.val
  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.neg

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_a_safe_fp_math:
; SI:  v_sub_f32_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; SI:  v_mac_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A]]
; VI:  v_sub_f16_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A]]
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_neg_a_safe_fp_math(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %a.neg = fsub half 0.0, %a.val
  %t.val = fmul half %a.neg, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_b_safe_fp_math:
; SI:  v_sub_f32_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; SI:  v_mac_f32_e32 v{{[0-9]+}}, v[[NEG_A]], v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v[[NEG_A]], v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_neg_b_safe_fp_math(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %b.neg = fsub half 0.0, %b.val
  %t.val = fmul half %a.val, %b.neg
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_c_safe_fp_math:
; SI:  v_sub_f32_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; SI:  v_mac_f32_e32 v[[NEG_A]], v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v[[NEG_A]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_neg_c_safe_fp_math(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %c.neg = fsub half 0.0, %c.val
  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.neg

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_a_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_mad_f32 v{{[0-9]+}}, -[[CVT_A]], [[CVT_B]], [[CVT_C]]

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_a_nsz_fp_math(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #1 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %a.neg = fsub half 0.0, %a.val
  %t.val = fmul half %a.neg, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_b_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_mad_f32 v{{[0-9]+}}, -[[CVT_A]], [[CVT_B]], [[CVT_C]]

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_b_nsz_fp_math(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #1 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %b.neg = fsub half 0.0, %b.val
  %t.val = fmul half %a.val, %b.neg
  %r.val = fadd half %t.val, %c.val

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_c_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_mad_f32 v{{[0-9]+}}, [[CVT_A]], [[CVT_B]], -[[CVT_C]]

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_c_nsz_fp_math(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) #1 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %c.neg = fsub half 0.0, %c.val
  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.neg

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16:
; GCN: {{buffer|flat}}_load_dword v[[A_V2_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_dword v[[B_V2_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_dword v[[C_V2_F16:[0-9]+]]

; SI:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[C_F32_0:[0-9]+]], v[[C_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI:  v_cvt_f32_f16_e32 v[[C_F32_1:[0-9]+]], v[[C_F16_1]]
; SI-DAG:  v_mac_f32_e32 v[[C_F32_0]], v[[B_F32_0]], v[[A_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_LO:[0-9]+]], v[[C_F32_0]]
; SI-DAG:  v_mac_f32_e32 v[[C_F32_1]], v[[B_F32_1]], v[[A_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[C_F32_1]]
; SI:  v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]

; VI: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; VI: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI:  v_mac_f16_e32 v[[C_V2_F16]], v[[B_V2_F16]], v[[A_V2_F16]]
; VI:  v_mac_f16_e32 v[[C_F16_1]], v[[B_F16_1]], v[[A_F16_1]]
; VI:  v_and_b32_e32 v[[R_F16_LO:[0-9]+]], 0xffff, v[[C_V2_F16]]
; VI:  v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[C_F16_1]]
; GCN: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_HI]], v[[R_F16_LO]]
; GCN: {{buffer|flat}}_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_same_add:
; SI:  v_mad_f32 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SI:  v_mad_f32 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SI:  v_mac_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SI:  v_mac_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_mad_f16 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD0:v[0-9]+]]
; VI:  v_mad_f16 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD1:v[0-9]+]]
; VI:  v_mac_f16_e32 [[ADD0]], v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_mac_f16_e32 [[ADD1]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_same_add(
    <2 x half> addrspace(1)* %r0,
    <2 x half> addrspace(1)* %r1,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c,
    <2 x half> addrspace(1)* %d,
    <2 x half> addrspace(1)* %e) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c
  %d.val = load <2 x half>, <2 x half> addrspace(1)* %d
  %e.val = load <2 x half>, <2 x half> addrspace(1)* %e

  %t0.val = fmul <2 x half> %a.val, %b.val
  %r0.val = fadd <2 x half> %t0.val, %c.val

  %t1.val = fmul <2 x half> %d.val, %e.val
  %r1.val = fadd <2 x half> %t1.val, %c.val

  store <2 x half> %r0.val, <2 x half> addrspace(1)* %r0
  store <2 x half> %r1.val, <2 x half> addrspace(1)* %r1
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_a:
; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT1:v[0-9]+]], {{v[0-9]+}}

; SI-DAG: v_mad_f32 v{{[0-9]+}}, -[[CVT0]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-DAG: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, [[CVT1]], v{{[0-9]+}}

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_a(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %a.neg = fsub <2 x half> <half -0.0, half -0.0>, %a.val
  %t.val = fmul <2 x half> %a.neg, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_b
; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT1:v[0-9]+]], {{v[0-9]+}}
; SI-DAG: v_mad_f32 v{{[0-9]+}}, -[[CVT0]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-DAG: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, [[CVT1]], v{{[0-9]+}}


; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_b(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %b.neg = fsub <2 x half> <half -0.0, half -0.0>, %b.val
  %t.val = fmul <2 x half> %a.val, %b.neg
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_c:
; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT1:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT2:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT3:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT4:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT5:v[0-9]+]], {{v[0-9]+}}

; SI-DAG: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -[[CVT2]]
; SI-DAG: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -[[CVT5]]

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_c(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %c.neg = fsub <2 x half> <half -0.0, half -0.0>, %c.val
  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.neg

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_a_safe_fp_math:
; SI:  v_sub_f32_e32 v[[NEG_A0:[0-9]+]], 0, v{{[0-9]+}}
; SI:  v_sub_f32_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; SI-DAG:  v_mac_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A0]]
; SI-DAG:  v_mac_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A1]]
; VI:  v_sub_f16_e32 v[[NEG_A0:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A0]]
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A1]]
; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_a_safe_fp_math(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %a.neg = fsub <2 x half> <half 0.0, half 0.0>, %a.val
  %t.val = fmul <2 x half> %a.neg, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_b_safe_fp_math:
; SI:  v_sub_f32_e32 v[[NEG_A0:[0-9]+]], 0, v{{[0-9]+}}
; SI:  v_sub_f32_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; SI-DAG:  v_mac_f32_e32 v{{[0-9]+}}, v[[NEG_A0]], v{{[0-9]+}}
; SI-DAG:  v_mac_f32_e32 v{{[0-9]+}}, v[[NEG_A1]], v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A0:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v[[NEG_A0]], v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v[[NEG_A1]], v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_b_safe_fp_math(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %b.neg = fsub <2 x half> <half 0.0, half 0.0>, %b.val
  %t.val = fmul <2 x half> %a.val, %b.neg
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_c_safe_fp_math:
; SI:  v_sub_f32_e32 v[[NEG_A0:[0-9]+]], 0, v{{[0-9]+}}
; SI:  v_sub_f32_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; SI-DAG:  v_mac_f32_e32 v[[NEG_A0]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-DAG:  v_mac_f32_e32 v[[NEG_A1]], v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A0:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_sub_f16_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v[[NEG_A0]], v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v[[NEG_A1]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_c_safe_fp_math(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #0 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %c.neg = fsub <2 x half> <half 0.0, half 0.0>, %c.val
  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.neg

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_a_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT1:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT2:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT3:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT4:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT5:v[0-9]+]], {{v[0-9]+}}

; SI-DAG: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SI-DAG: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_a_nsz_fp_math(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #1 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %a.neg = fsub <2 x half> <half 0.0, half 0.0>, %a.val
  %t.val = fmul <2 x half> %a.neg, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_b_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT1:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT2:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT3:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT4:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT5:v[0-9]+]], {{v[0-9]+}}

; SI-DAG: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SI-DAG: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_b_nsz_fp_math(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #1 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %b.neg = fsub <2 x half> <half 0.0, half 0.0>, %b.val
  %t.val = fmul <2 x half> %a.val, %b.neg
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_c_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT1:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT2:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT3:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT4:v[0-9]+]], {{v[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT5:v[0-9]+]], {{v[0-9]+}}

; SI-DAG: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; SI-DAG: v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[-0-9]}}
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[-0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_c_nsz_fp_math(
    <2 x half> addrspace(1)* %r,
    <2 x half> addrspace(1)* %a,
    <2 x half> addrspace(1)* %b,
    <2 x half> addrspace(1)* %c) #1 {
entry:
  %a.val = load <2 x half>, <2 x half> addrspace(1)* %a
  %b.val = load <2 x half>, <2 x half> addrspace(1)* %b
  %c.val = load <2 x half>, <2 x half> addrspace(1)* %c

  %c.neg = fsub <2 x half> <half 0.0, half 0.0>, %c.val
  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.neg

  store <2 x half> %r.val, <2 x half> addrspace(1)* %r
  ret void
}

attributes #0 = { nounwind "no-signed-zeros-fp-math"="false" }
attributes #1 = { nounwind "no-signed-zeros-fp-math"="true" }
