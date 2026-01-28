; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -mcpu=tahiti < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -mcpu=fiji -mattr=-flat-for-global < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=VI %s

; FIXME: Can the SI case form the mac through the casts?

; GCN-LABEL: {{^}}mac_f16:
; GCN: {{buffer|flat}}_load_ushort v[[A_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort v[[B_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort v[[C_F16:[0-9]+]]

; SI:  v_mul_f32_e32
; SI:  v_cvt_f16_f32
; SI:  v_cvt_f32_f16
; SI:  v_cvt_f32_f16
; SI:  v_add_f32_e32

; VI:  v_mac_f16_e32 v[[C_F16]], v[[A_F16]], v[[B_F16]]
; VI:  buffer_store_short v[[C_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_same_add:
; SI:  v_mul_f32_e32
; SI:  v_mul_f32_e32
; SI:  v_cvt_f16_f32
; SI:  v_cvt_f16_f32
; SI:  v_cvt_f32_f16
; SI:  v_cvt_f32_f16
; SI:  v_add_f32_e32
; SI:  v_add_f32_e32

; VI:  v_mad_f16 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, [[ADD:v[0-9]+]]
; VI:  v_mac_f16_e32 [[ADD]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_same_add(
    ptr addrspace(1) %r0,
    ptr addrspace(1) %r1,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c,
    ptr addrspace(1) %d,
    ptr addrspace(1) %e) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c
  %d.val = load half, ptr addrspace(1) %d
  %e.val = load half, ptr addrspace(1) %e

  %t0.val = fmul half %a.val, %b.val
  %r0.val = fadd half %t0.val, %c.val

  %t1.val = fmul half %d.val, %e.val
  %r1.val = fadd half %t1.val, %c.val

  store half %r0.val, ptr addrspace(1) %r0
  store half %r1.val, ptr addrspace(1) %r1
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_a:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16
; SI: v_sub_f32


; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_a(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %a.neg = fneg half %a.val
  %t.val = fmul half %a.neg, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_b:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_mul_f32_e32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16
; SI: v_sub_f32_e32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %b.neg = fneg half %b.val
  %t.val = fmul half %a.val, %b.neg
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_c:
; SI: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; SI: v_mul_f32_e32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; SI: v_sub_f32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_c(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %c.neg = fneg half %c.val
  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.neg

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_a_safe_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_cvt_f16_f32_e64 v{{[0-9]+}}, -v{{[0-9]+}}

; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16
; SI: v_add_f32
; SI: v_cvt_f16_f32

; VI:  v_sub_f16_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v[[NEG_A]], v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_neg_a_safe_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %a.neg = fsub half 0.0, %a.val
  %t.val = fmul half %a.neg, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_b_safe_fp_math:

; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32_e32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16
; SI: v_add_f32_e32
; SI: v_cvt_f16_f32

; VI:  v_sub_f16_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A]]
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_neg_b_safe_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %b.neg = fsub half 0.0, %b.val
  %t.val = fmul half %a.val, %b.neg
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_c_safe_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32_e32
; SI: v_cvt_f16_f32_e64 v{{[0-9]+}}, -v{{[0-9]}}
; SI: v_add_f32_e32
; SI: v_cvt_f16_f32

; VI:  v_sub_f16_e32 v[[NEG_A:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v[[NEG_A]], v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @mac_f16_neg_c_safe_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %c.neg = fsub half 0.0, %c.val
  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.neg

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_a_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_mul_f32_e32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16
; SI: v_sub_f32_e32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_a_nsz_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #1 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %a.neg = fsub nsz half 0.0, %a.val
  %t.val = fmul half %a.neg, %b.val
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_b_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_sub_f32_e32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_b_nsz_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #1 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %b.neg = fsub nsz half 0.0, %b.val
  %t.val = fmul half %a.val, %b.neg
  %r.val = fadd half %t.val, %c.val

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_f16_neg_c_nsz_fp_math:
; SI: v_cvt_f32_f16_e32 [[CVT_A:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16_e32 [[CVT_B:v[0-9]+]], v{{[0-9]+}}
; SI: v_mul_f32_e32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16_e32 [[CVT_C:v[0-9]+]], v{{[0-9]+}}
; SI: v_cvt_f32_f16
; SI: v_sub_f32_e32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_f16_neg_c_nsz_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #1 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %c.val = load half, ptr addrspace(1) %c

  %c.neg = fsub nsz half 0.0, %c.val
  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, %c.neg

  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16:
; SI: v_mul_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16
; SI: v_add_f32
; SI: v_add_f32

; VI: {{buffer|flat}}_load_dword v[[A_V2_F16:[0-9]+]]
; VI: {{buffer|flat}}_load_dword v[[B_V2_F16:[0-9]+]]
; VI: {{buffer|flat}}_load_dword v[[C_V2_F16:[0-9]+]]
; VI-DAG: v_lshrrev_b32_e32 v[[C_F16_1:[0-9]+]], 16, v[[C_V2_F16]]
; VI-DAG: v_mac_f16_sdwa v[[C_F16_1]], v[[A_V2_F16]], v[[B_V2_F16]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG: v_mac_f16_e32 v[[C_V2_F16]], v[[A_V2_F16]], v[[B_V2_F16]]
; VI-DAG: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[C_F16_1]]
; VI-NOT: and
; VI:  v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[C_V2_F16]], v[[R_F16_HI]]

; VI: {{buffer|flat}}_store_dword v[[R_V2_F16]]
; VI: s_endpgm
define amdgpu_kernel void @mac_v2f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  call void @llvm.amdgcn.s.barrier() #2
  %b.val = load <2 x half>, ptr addrspace(1) %b
  call void @llvm.amdgcn.s.barrier() #2
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_same_add:
; SI: v_mul_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_cvt_f16_f32
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_add_f32
; SI: v_add_f32

; VI-DAG:  v_mac_f16_sdwa v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG:  v_mad_f16 v{{[0-9]}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG:  v_mac_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_same_add(
    ptr addrspace(1) %r0,
    ptr addrspace(1) %r1,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c,
    ptr addrspace(1) %d,
    ptr addrspace(1) %e) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c
  %d.val = load <2 x half>, ptr addrspace(1) %d
  %e.val = load <2 x half>, ptr addrspace(1) %e

  %t0.val = fmul <2 x half> %a.val, %b.val
  %r0.val = fadd <2 x half> %t0.val, %c.val

  %t1.val = fmul <2 x half> %d.val, %e.val
  %r1.val = fadd <2 x half> %t1.val, %c.val

  store <2 x half> %r0.val, ptr addrspace(1) %r0
  store <2 x half> %r1.val, ptr addrspace(1) %r1
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_a:
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_a(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %a.neg = fneg <2 x half> %a.val
  %t.val = fmul <2 x half> %a.neg, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_b
; SI: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32_e32
; SI: v_cvt_f16_f32
; SI: v_sub_f32_e32


; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %b.neg = fneg <2 x half> %b.val
  %t.val = fmul <2 x half> %a.val, %b.neg
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_c:
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f32_f16

; SI: v_sub_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[0-9]+}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_c(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %c.neg = fneg <2 x half> %c.val
  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.neg

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_a_safe_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f32_f16
; SI: v_add_f32
; SI: v_cvt_f16_f32
; SI: v_add_f32
; SI: v_cvt_f16_f32

; VI-DAG:  v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; VI-DAG:  v_sub_f16_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; VI-DAG:  v_sub_f16_sdwa v[[NEG_A0:[0-9]+]], [[ZERO]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-DAG:  v_mac_f16_sdwa v{{[0-9]+}}, v[[NEG_A0]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-DAG:  v_mac_f16_e32 v{{[0-9]+}}, v[[NEG_A1]], v{{[0-9]+}}

; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_a_safe_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %a.neg = fsub <2 x half> <half 0.0, half 0.0>, %a.val
  %t.val = fmul <2 x half> %a.neg, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_b_safe_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_add_f32
; SI: v_add_f32

; VI:  v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; VI:  v_sub_f16_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_sub_f16_sdwa v[[NEG_A0:[0-9]+]], [[ZERO]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-DAG:  v_mac_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A0]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v[[NEG_A1]]

; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_b_safe_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %b.neg = fsub <2 x half> <half 0.0, half 0.0>, %b.val
  %t.val = fmul <2 x half> %a.val, %b.neg
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_c_safe_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32
; SI: v_mul_f32
; SI: v_add_f32
; SI: v_add_f32

; VI:  v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; VI:  v_sub_f16_e32 v[[NEG_A1:[0-9]+]], 0, v{{[0-9]+}}
; VI:  v_sub_f16_sdwa v[[NEG_A0:[0-9]+]], [[ZERO]], v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
; VI-DAG:  v_mac_f16_sdwa v[[NEG_A0]], v{{[0-9]+}}, v{{[0-9]+}} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI-DAG:  v_mac_f16_e32 v[[NEG_A1]], v{{[0-9]+}}, v{{[0-9]+}}

; GCN: s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_c_safe_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #0 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %c.neg = fsub <2 x half> <half 0.0, half 0.0>, %c.val
  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.neg

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_a_nsz_fp_math:
; SI: v_cvt_f32_f16_e32
; SI: v_cvt_f32_f16_e32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_a_nsz_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #1 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %a.neg = fsub nsz <2 x half> <half 0.0, half 0.0>, %a.val
  %t.val = fmul <2 x half> %a.neg, %b.val
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_b_nsz_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; VI:     v_mad_f16 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[-0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_b_nsz_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #1 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %b.neg = fsub nsz <2 x half> <half 0.0, half 0.0>, %b.val
  %t.val = fmul <2 x half> %a.val, %b.neg
  %r.val = fadd <2 x half> %t.val, %c.val

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}mac_v2f16_neg_c_nsz_fp_math:
; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_mul_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32
; SI: v_sub_f32
; SI: v_cvt_f16_f32

; VI-NOT: v_mac_f16
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[-0-9]}}
; VI:     v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, -v{{[-0-9]}}
; GCN:    s_endpgm
define amdgpu_kernel void @mac_v2f16_neg_c_nsz_fp_math(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(1) %c) #1 {
entry:
  %a.val = load <2 x half>, ptr addrspace(1) %a
  %b.val = load <2 x half>, ptr addrspace(1) %b
  %c.val = load <2 x half>, ptr addrspace(1) %c

  %c.neg = fsub nsz <2 x half> <half 0.0, half 0.0>, %c.val
  %t.val = fmul <2 x half> %a.val, %b.val
  %r.val = fadd <2 x half> %t.val, %c.neg

  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

declare void @llvm.amdgcn.s.barrier() #2

attributes #0 = { nounwind "no-signed-zeros-fp-math"="false" "denormal-fp-math"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind "denormal-fp-math"="preserve-sign,preserve-sign" }
attributes #2 = { nounwind convergent }
