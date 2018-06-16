; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -O0 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}eq_t:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_cmp_lt_f32_e{{32|64}} [[CC:s\[[0-9]+:[0-9]+\]|vcc]], [[X]], [[ONE]]{{$}}
; GCN-NOT: 0xddd5
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_cmp_eq_u32
; GCN-NOT: v_cndmask_b32
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[FOUR:v[0-9]+]], 4.0
; GCN:     v_cndmask_b32_e{{32|64}} [[RES:v[0-9]+]], [[TWO]], [[FOUR]], [[CC]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @eq_t(float %x) {
  %c1 = fcmp olt float %x, 1.0
  %s1 = select i1 %c1, i32 56789, i32 1
  %c2 = icmp eq i32 %s1, 56789
  %s2 = select i1 %c2, float 4.0, float 2.0
  store float %s2, float* undef, align 4
  ret void
}

; GCN-LABEL: {{^}}ne_t:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_cmp_lt_f32_e{{32|64}} [[CC:s\[[0-9]+:[0-9]+\]|vcc]], [[X]], [[ONE]]{{$}}
; GCN-NOT: 0xddd5
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_cmp_eq_u32
; GCN-NOT: v_cndmask_b32
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[FOUR:v[0-9]+]], 4.0
; GCN:     v_cndmask_b32_e{{32|64}} [[RES:v[0-9]+]], [[FOUR]], [[TWO]], [[CC]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @ne_t(float %x) {
  %c1 = fcmp olt float %x, 1.0
  %s1 = select i1 %c1, i32 56789, i32 1
  %c2 = icmp ne i32 %s1, 56789
  %s2 = select i1 %c2, float 4.0, float 2.0
  store float %s2, float* undef, align 4
  ret void
}

; GCN-LABEL: {{^}}eq_f:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_cmp_lt_f32_e{{32|64}} [[CC:s\[[0-9]+:[0-9]+\]|vcc]], [[X]], [[ONE]]{{$}}
; GCN-NOT: 0xddd5
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_cmp_eq_u32
; GCN-NOT: v_cndmask_b32
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[FOUR:v[0-9]+]], 4.0
; GCN:     v_cndmask_b32_e{{32|64}} [[RES:v[0-9]+]], [[FOUR]], [[TWO]], [[CC]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @eq_f(float %x) {
  %c1 = fcmp olt float %x, 1.0
  %s1 = select i1 %c1, i32 1, i32 56789
  %c2 = icmp eq i32 %s1, 56789
  %s2 = select i1 %c2, float 4.0, float 2.0
  store float %s2, float* undef, align 4
  ret void
}

; GCN-LABEL: {{^}}ne_f:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_cmp_lt_f32_e{{32|64}} [[CC:s\[[0-9]+:[0-9]+\]|vcc]], [[X]], [[ONE]]{{$}}
; GCN-NOT: 0xddd5
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_cmp_eq_u32
; GCN-NOT: v_cndmask_b32
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[FOUR:v[0-9]+]], 4.0
; GCN:     v_cndmask_b32_e{{32|64}} [[RES:v[0-9]+]], [[TWO]], [[FOUR]], [[CC]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @ne_f(float %x) {
  %c1 = fcmp olt float %x, 1.0
  %s1 = select i1 %c1, i32 1, i32 56789
  %c2 = icmp ne i32 %s1, 56789
  %s2 = select i1 %c2, float 4.0, float 2.0
  store float %s2, float* undef, align 4
  ret void
}

; GCN-LABEL: {{^}}different_constants:
; GCN-DAG: s_load_dword [[X:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN-DAG: v_cmp_lt_f32_e{{32|64}} [[CC1:s\[[0-9]+:[0-9]+\]|vcc]], [[X]], [[ONE]]{{$}}
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[CND1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_cmp_eq_u32_e{{32|64}} [[CC2:s\[[0-9]+:[0-9]+\]|vcc]], v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[FOUR:v[0-9]+]], 4.0
; GCN:     v_cndmask_b32_e{{32|64}} [[RES:v[0-9]+]], [[TWO]], [[FOUR]], [[CC2]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]{{$}}
define amdgpu_kernel void @different_constants(float %x) {
  %c1 = fcmp olt float %x, 1.0
  %s1 = select i1 %c1, i32 56789, i32 1
  %c2 = icmp eq i32 %s1, 5678
  %s2 = select i1 %c2, float 4.0, float 2.0
  store float %s2, float* undef, align 4
  ret void
}
