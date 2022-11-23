; RUN: llc -march=amdgcn -verify-machineinstrs < %s| FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; XXX: Merge this into setcc, once R600 supports 64-bit operations

;;;==========================================================================;;;
;; Double comparisons
;;;==========================================================================;;;

; GCN-LABEL: {{^}}f64_oeq:
; GCN: v_cmp_eq_f64
define amdgpu_kernel void @f64_oeq(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp oeq double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ogt:
; GCN: v_cmp_gt_f64
define amdgpu_kernel void @f64_ogt(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ogt double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_oge:
; GCN: v_cmp_ge_f64
define amdgpu_kernel void @f64_oge(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp oge double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_olt:
; GCN: v_cmp_lt_f64
define amdgpu_kernel void @f64_olt(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp olt double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ole:
; GCN: v_cmp_le_f64
define amdgpu_kernel void @f64_ole(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ole double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_one:
; GCN: v_cmp_lg_f64_e32 vcc
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f64_one(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp one double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ord:
; GCN: v_cmp_o_f64
define amdgpu_kernel void @f64_ord(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ord double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ueq:
; GCN: v_cmp_nlg_f64_e32 vcc
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f64_ueq(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ueq double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ugt:

; GCN: v_cmp_nle_f64_e32 vcc
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f64_ugt(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ugt double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_uge:
; GCN: v_cmp_nlt_f64_e32 vcc
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f64_uge(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp uge double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ult:
; GCN: v_cmp_nge_f64_e32 vcc
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f64_ult(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ult double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_ule:
; GCN: v_cmp_ngt_f64_e32 vcc
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, 0, -1, vcc
define amdgpu_kernel void @f64_ule(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp ule double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_une:
; GCN: v_cmp_neq_f64
define amdgpu_kernel void @f64_une(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp une double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}f64_uno:
; GCN: v_cmp_u_f64
define amdgpu_kernel void @f64_uno(i32 addrspace(1)* %out, double %a, double %b) #0 {
entry:
  %tmp0 = fcmp uno double %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

;;;==========================================================================;;;
;; 64-bit integer comparisons
;;;==========================================================================;;;

; GCN-LABEL: {{^}}i64_eq:
; SI: v_cmp_eq_u64
; VI: s_cmp_eq_u64
define amdgpu_kernel void @i64_eq(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp eq i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_ne:
; SI: v_cmp_ne_u64
; VI: s_cmp_lg_u64
define amdgpu_kernel void @i64_ne(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp ne i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_ugt:
; GCN: v_cmp_gt_u64
define amdgpu_kernel void @i64_ugt(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp ugt i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_uge:
; GCN: v_cmp_ge_u64
define amdgpu_kernel void @i64_uge(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp uge i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_ult:
; GCN: v_cmp_lt_u64
define amdgpu_kernel void @i64_ult(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp ult i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_ule:
; GCN: v_cmp_le_u64
define amdgpu_kernel void @i64_ule(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp ule i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_sgt:
; GCN: v_cmp_gt_i64
define amdgpu_kernel void @i64_sgt(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp sgt i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_sge:
; GCN: v_cmp_ge_i64
define amdgpu_kernel void @i64_sge(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp sge i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_slt:
; GCN: v_cmp_lt_i64
define amdgpu_kernel void @i64_slt(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp slt i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i64_sle:
; GCN: v_cmp_le_i64
define amdgpu_kernel void @i64_sle(i32 addrspace(1)* %out, i64 %a, i64 %b) #0 {
entry:
  %tmp0 = icmp sle i64 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i128_sle:
; GCN: v_cmp_le_i64
; CGV: v_cndmask
; SI: v_cmp_eq_u64
; VI: s_cmp_eq_u64
define amdgpu_kernel void @i128_sle(i32 addrspace(1)* %out, i128 %a, i128 %b) #0 {
entry:
  %tmp0 = icmp sle i128 %a, %b
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}i128_eq_const:
; SI: v_cmp_eq_u64
; VI: s_cmp_eq_u64
define amdgpu_kernel void @i128_eq_const(i32 addrspace(1)* %out, i128 %a) #0 {
entry:
  %tmp0 = icmp eq i128 %a, 85070591730234615865843651857942052992
  %tmp1 = sext i1 %tmp0 to i32
  store i32 %tmp1, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
