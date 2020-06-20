; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Make sure to test with f32 and i32 compares. If we have to use float
; compares, we always have multiple condition registers. If we can do
; scalar compares, we don't want to use multiple condition registers.

; GCN-LABEL: {{^}}opt_select_i32_and_cmp_i32:
; GCN-DAG: v_cmp_ne_u32_e32 vcc,
; GCN-DAG: v_cmp_ne_u32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_and_b64 [[CMP1]], vcc, [[CMP1]]
; GCN: s_cselect_b32 [[SRESULT:s[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GCN-NOT: [[SRESULT]]
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], [[SRESULT]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @opt_select_i32_and_cmp_i32(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) #0 {
  %icmp0 = icmp ne i32 %a, %b
  %icmp1 = icmp ne i32 %a, %c
  %and = and i1 %icmp0, %icmp1
  %select = select i1 %and, i32 %x, i32 %y
  store i32 %select, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i32_and_cmp_f32:
; GCN-DAG: v_cmp_lg_f32_e32 vcc
; GCN-DAG: v_cmp_lg_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_and_b64 [[CMP1]], vcc, [[CMP1]]
; GCN: s_cselect_b32 [[SRESULT:s[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GCN-NOT: [[SRESULT]]
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], [[SRESULT]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @opt_select_i32_and_cmp_f32(i32 addrspace(1)* %out, float %a, float %b, float %c, i32 %x, i32 %y) #0 {
  %fcmp0 = fcmp one float %a, %b
  %fcmp1 = fcmp one float %a, %c
  %and = and i1 %fcmp0, %fcmp1
  %select = select i1 %and, i32 %x, i32 %y
  store i32 %select, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i64_and_cmp_i32:
; GCN-DAG: v_cmp_ne_u32_e32 vcc,
; GCN-DAG: v_cmp_ne_u32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_and_b64 vcc, vcc, [[CMP1]]
; GCN: v_cndmask_b32_e32 v[[RESULT1:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v[[RESULT0:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT0]]:[[RESULT1]]{{\]}}
define amdgpu_kernel void @opt_select_i64_and_cmp_i32(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i64 %x, i64 %y) #0 {
  %icmp0 = icmp ne i32 %a, %b
  %icmp1 = icmp ne i32 %a, %c
  %and = and i1 %icmp0, %icmp1
  %select = select i1 %and, i64 %x, i64 %y
  store i64 %select, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i64_and_cmp_f32:
; GCN-DAG: v_cmp_lg_f32_e32 vcc,
; GCN-DAG: v_cmp_lg_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_and_b64 vcc, vcc, [[CMP1]]
; GCN: v_cndmask_b32_e32 v[[RESULT1:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v[[RESULT0:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT0]]:[[RESULT1]]{{\]}}
define amdgpu_kernel void @opt_select_i64_and_cmp_f32(i64 addrspace(1)* %out, float %a, float %b, float %c, i64 %x, i64 %y) #0 {
  %fcmp0 = fcmp one float %a, %b
  %fcmp1 = fcmp one float %a, %c
  %and = and i1 %fcmp0, %fcmp1
  %select = select i1 %and, i64 %x, i64 %y
  store i64 %select, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i32_or_cmp_i32:
; GCN-DAG: v_cmp_ne_u32_e32 vcc,
; GCN-DAG: v_cmp_ne_u32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_or_b64 [[CMP1]], vcc, [[CMP1]]
; GCN: s_cselect_b32 [[SRESULT:s[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GCN-NOT: [[SRESULT]]
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], [[SRESULT]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @opt_select_i32_or_cmp_i32(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) #0 {
  %icmp0 = icmp ne i32 %a, %b
  %icmp1 = icmp ne i32 %a, %c
  %or = or i1 %icmp0, %icmp1
  %select = select i1 %or, i32 %x, i32 %y
  store i32 %select, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i32_or_cmp_f32:
; GCN-DAG: v_cmp_lg_f32_e32 vcc
; GCN-DAG: v_cmp_lg_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_or_b64 [[CMP1]], vcc, [[CMP1]]
; GCN: s_cselect_b32 [[SRESULT:s[0-9]+]], {{s[0-9]+}}, {{s[0-9]+}}
; GCN-NOT: [[SRESULT]]
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], [[SRESULT]]
; GCN-NOT: [[RESULT]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @opt_select_i32_or_cmp_f32(i32 addrspace(1)* %out, float %a, float %b, float %c, i32 %x, i32 %y) #0 {
  %fcmp0 = fcmp one float %a, %b
  %fcmp1 = fcmp one float %a, %c
  %or = or i1 %fcmp0, %fcmp1
  %select = select i1 %or, i32 %x, i32 %y
  store i32 %select, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i64_or_cmp_i32:
; GCN-DAG: v_cmp_ne_u32_e32 vcc,
; GCN-DAG: v_cmp_ne_u32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_or_b64 vcc, vcc, [[CMP1]]
; GCN: v_cndmask_b32_e32 v[[RESULT1:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v[[RESULT0:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT0]]:[[RESULT1]]{{\]}}
define amdgpu_kernel void @opt_select_i64_or_cmp_i32(i64 addrspace(1)* %out, i32 %a, i32 %b, i32 %c, i64 %x, i64 %y) #0 {
  %icmp0 = icmp ne i32 %a, %b
  %icmp1 = icmp ne i32 %a, %c
  %or = or i1 %icmp0, %icmp1
  %select = select i1 %or, i64 %x, i64 %y
  store i64 %select, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}opt_select_i64_or_cmp_f32:
; GCN-DAG: v_cmp_lg_f32_e32 vcc,
; GCN-DAG: v_cmp_lg_f32_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]]
; GCN: s_or_b64 vcc, vcc, [[CMP1]]
; GCN: v_cndmask_b32_e32 v[[RESULT1:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v[[RESULT0:[0-9]+]], {{v[0-9]+}}, {{v[0-9]+}}, vcc
; GCN: buffer_store_dwordx2 v{{\[}}[[RESULT0]]:[[RESULT1]]{{\]}}
define amdgpu_kernel void @opt_select_i64_or_cmp_f32(i64 addrspace(1)* %out, float %a, float %b, float %c, i64 %x, i64 %y) #0 {
  %fcmp0 = fcmp one float %a, %b
  %fcmp1 = fcmp one float %a, %c
  %or = or i1 %fcmp0, %fcmp1
  %select = select i1 %or, i64 %x, i64 %y
  store i64 %select, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}regression:
; GCN: v_cmp_neq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}, 1.0
; GCN: v_cmp_neq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}, 0
; GCN: v_cmp_eq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}, 0

define amdgpu_kernel void @regression(float addrspace(1)* %out, float %c0, float %c1) #0 {
entry:
  %cmp0 = fcmp oeq float %c0, 1.0
  br i1 %cmp0, label %if0, label %endif

if0:
  %cmp1 = fcmp oeq float %c1, 0.0
  br i1 %cmp1, label %if1, label %endif

if1:
  %cmp2 = xor i1 %cmp1, true
  br label %endif

endif:
  %tmp0 = phi i1 [ true, %entry ], [ %cmp2, %if1 ], [ false, %if0 ]
  %tmp2 = select i1 %tmp0, float 4.0, float 0.0
  store float %tmp2, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
