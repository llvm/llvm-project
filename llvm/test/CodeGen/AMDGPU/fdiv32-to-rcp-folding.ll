; RUN: llc -march=amdgcn -mcpu=gfx900 -denormal-fp-math-f32=ieee < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-DENORM %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -denormal-fp-math-f32=preserve-sign < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-FLUSH %s

; GCN-LABEL: {{^}}div_1_by_x_25ulp:
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DAG:        s_load_dword [[VAL:s[0-9]+]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |[[VAL]]|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 [[SCALE:v[0-9]+]], 1.0, [[S]], vcc
; GCN-DENORM:     v_mul_f32_e32 [[PRESCALED:v[0-9]+]], [[VAL]], [[SCALE]]
; GCN-DENORM:     v_rcp_f32_e32 [[RCP:v[0-9]+]], [[PRESCALED]]
; GCN-DENORM:     v_mul_f32_e32 [[OUT:v[0-9]+]], [[SCALE]], [[RCP]]

; GCN-FLUSH:      v_rcp_f32_e32 [[OUT:v[0-9]+]], [[VAL]]

; GCN:            global_store_dword v{{[0-9]+}}, [[OUT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_1_by_x_25ulp(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv float 1.000000e+00, %load, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_minus_1_by_x_25ulp:
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DAG:        s_load_dword [[VAL:s[0-9]+]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |[[VAL]]|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 [[SCALE:v[0-9]+]], 1.0, [[S]], vcc
; GCN-DENORM:     v_mul_f32_e64 [[PRESCALED:v[0-9]+]], [[VAL]], -[[SCALE]]
; GCN-DENORM:     v_rcp_f32_e32 [[RCP:v[0-9]+]], [[PRESCALED]]
; GCN-DENORM:     v_mul_f32_e32 [[OUT:v[0-9]+]], [[SCALE]], [[RCP]]

; GCN-FLUSH:      v_rcp_f32_e64 [[OUT:v[0-9]+]], -[[VAL]]

; GCN:            global_store_dword v{{[0-9]+}}, [[OUT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_minus_1_by_x_25ulp(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv float -1.000000e+00, %load, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_1_by_minus_x_25ulp:
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DAG:        s_load_dword [[VAL:s[0-9]+]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |[[VAL]]|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 [[SCALE:v[0-9]+]], 1.0, [[S]], vcc
; GCN-DENORM:     v_mul_f32_e64 [[PRESCALED:v[0-9]+]], -[[VAL]], [[SCALE]]
; GCN-DENORM:     v_rcp_f32_e32 [[RCP:v[0-9]+]], [[PRESCALED]]
; GCN-DENORM:     v_mul_f32_e32 [[OUT:v[0-9]+]], [[SCALE]], [[RCP]]

; GCN-FLUSH:      v_rcp_f32_e64 [[OUT:v[0-9]+]], -[[VAL]]

; GCN:            global_store_dword v{{[0-9]+}}, [[OUT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_1_by_minus_x_25ulp(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %neg = fneg float %load
  %div = fdiv float 1.000000e+00, %neg, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_minus_1_by_minus_x_25ulp:
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DAG:        s_load_dword [[VAL:s[0-9]+]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |[[VAL]]|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 [[SCALE:v[0-9]+]], 1.0, [[S]], vcc
; GCN-DENORM:     v_mul_f32_e32 [[PRESCALED:v[0-9]+]], [[VAL]], [[SCALE]]
; GCN-DENORM:     v_rcp_f32_e32 [[RCP:v[0-9]+]], [[PRESCALED]]
; GCN-DENORM:     v_mul_f32_e32 [[OUT:v[0-9]+]], [[SCALE]], [[RCP]]

; GCN-FLUSH:      v_rcp_f32_e32 [[OUT:v[0-9]+]], [[VAL]]

; GCN:            global_store_dword v{{[0-9]+}}, [[OUT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_minus_1_by_minus_x_25ulp(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %neg = fsub float -0.000000e+00, %load
  %div = fdiv float -1.000000e+00, %neg, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_v4_1_by_x_25ulp:
; GCN-DAG:        s_load_dwordx4 s[[[VAL0:[0-9]+]]:[[VAL3:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32

; GCN-FLUSH:      v_rcp_f32_e32 v[[OUT0:[0-9]+]], s[[VAL0]]
; GCN-FLUSH:      v_rcp_f32_e32
; GCN-FLUSH:      v_rcp_f32_e32
; GCN-FLUSH:      v_rcp_f32_e32 v[[OUT3:[0-9]+]], s[[VAL3]]
; GCN-FLUSH:      global_store_dwordx4 v{{[0-9]+}}, v[[[OUT0]]:[[OUT3]]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_v4_1_by_x_25ulp(<4 x float> addrspace(1)* %arg) {
  %load = load <4 x float>, <4 x float> addrspace(1)* %arg, align 16
  %div = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %load, !fpmath !0
  store <4 x float> %div, <4 x float> addrspace(1)* %arg, align 16
  ret void
}

; GCN-LABEL: {{^}}div_v4_minus_1_by_x_25ulp:
; GCN-DAG:        s_load_dwordx4 s[[[VAL0:[0-9]+]]:[[VAL3:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32

; GCN-FLUSH:      v_rcp_f32_e64 v[[OUT0:[0-9]+]], -s[[VAL0]]
; GCN-FLUSH:      v_rcp_f32_e64
; GCN-FLUSH:      v_rcp_f32_e64
; GCN-FLUSH:      v_rcp_f32_e64 v[[OUT3:[0-9]+]], -s[[VAL3]]
define amdgpu_kernel void @div_v4_minus_1_by_x_25ulp(<4 x float> addrspace(1)* %arg) {
  %load = load <4 x float>, <4 x float> addrspace(1)* %arg, align 16
  %div = fdiv <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>, %load, !fpmath !0
  store <4 x float> %div, <4 x float> addrspace(1)* %arg, align 16
  ret void
}

; GCN-LABEL: {{^}}div_v4_1_by_minus_x_25ulp:
; GCN-DAG:        s_load_dwordx4 s[[[VAL0:[0-9]+]]:[[VAL3:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32

; GCN-FLUSH:      v_rcp_f32_e64 v[[OUT0:[0-9]+]], -s[[VAL0]]
; GCN-FLUSH:      v_rcp_f32_e64
; GCN-FLUSH:      v_rcp_f32_e64
; GCN-FLUSH:      v_rcp_f32_e64 v[[OUT3:[0-9]+]], -s[[VAL3]]
; GCN-FLUSH:      global_store_dwordx4 v{{[0-9]+}}, v[[[OUT0]]:[[OUT3]]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_v4_1_by_minus_x_25ulp(<4 x float> addrspace(1)* %arg) {
  %load = load <4 x float>, <4 x float> addrspace(1)* %arg, align 16
  %neg = fneg <4 x float> %load
  %div = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %neg, !fpmath !0
  store <4 x float> %div, <4 x float> addrspace(1)* %arg, align 16
  ret void
}

; GCN-LABEL: {{^}}div_v4_minus_1_by_minus_x_25ulp:
; GCN-DAG:        s_load_dwordx4 s[[[VAL0:[0-9]+]]:[[VAL3:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; GCN-DENORM-DAG: v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DENORM-DAG: v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DENORM-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32
; GCN-DENORM-DAG: v_mul_f32_e32

; GCN-FLUSH:      v_rcp_f32_e32 v[[OUT0:[0-9]+]], s[[VAL0]]
; GCN-FLUSH:      v_rcp_f32_e32
; GCN-FLUSH:      v_rcp_f32_e32
; GCN-FLUSH:      v_rcp_f32_e32 v[[OUT3:[0-9]+]], s[[VAL3]]
; GCN-FLUSH:      global_store_dwordx4 v{{[0-9]+}}, v[[[OUT0]]:[[OUT3]]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_v4_minus_1_by_minus_x_25ulp(<4 x float> addrspace(1)* %arg) {
  %load = load <4 x float>, <4 x float> addrspace(1)* %arg, align 16
  %neg = fneg <4 x float> %load
  %div = fdiv <4 x float> <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>, %neg, !fpmath !0
  store <4 x float> %div, <4 x float> addrspace(1)* %arg, align 16
  ret void
}

; GCN-LABEL: {{^}}div_v4_c_by_x_25ulp:
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, 2.0{{$}}
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, 2.0{{$}}
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32

; GCN-DAG:        v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DAG:        v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000

; GCN-DAG:        v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DAG:        v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DAG:        v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DAG:        v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc

; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -v{{[0-9]+}}
; GCN-DENORM-DAG: v_rcp_f32_e32 [[RCP1:v[0-9]+]], v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[RCP1]]
; GCN-DENORM-DAG: v_rcp_f32_e32 [[RCP2:v[0-9]+]], v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[RCP2]]

; GCN-DENORM-DAG: v_div_fmas_f32
; GCN-DENORM-DAG: v_div_fmas_f32
; GCN-DENORM-DAG: v_div_fixup_f32 {{.*}}, 2.0{{$}}
; GCN-DENORM-DAG: v_div_fixup_f32 {{.*}}, -2.0{{$}}

; GCN-FLUSH-DAG:  v_rcp_f32_e32
; GCN-FLUSH-DAG:  v_rcp_f32_e64

; GCN-NOT:        v_cmp_gt_f32_e64
; GCN-NOT:        v_cndmask_b32_e32
; GCN-FLUSH-NOT:  v_div

; GCN:            global_store_dwordx4
define amdgpu_kernel void @div_v4_c_by_x_25ulp(<4 x float> addrspace(1)* %arg) {
  %load = load <4 x float>, <4 x float> addrspace(1)* %arg, align 16
  %div = fdiv <4 x float> <float 2.000000e+00, float 1.000000e+00, float -1.000000e+00, float -2.000000e+00>, %load, !fpmath !0
  store <4 x float> %div, <4 x float> addrspace(1)* %arg, align 16
  ret void
}

; GCN-LABEL: {{^}}div_v4_c_by_minus_x_25ulp:
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_div_scale_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_rcp_f32_e32

; GCN-DAG:        v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-DAG:        v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000

; GCN-DAG:        v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DAG:        v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc
; GCN-DAG:        v_cmp_gt_f32_e64 vcc, |s{{[0-9]+}}|, [[L]]
; GCN-DAG:        v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, [[S]], vcc

; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e64 v{{[0-9]+}}, -s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DENORM-DAG: v_rcp_f32_e32 [[RCP1:v[0-9]+]], v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[RCP1]]
; GCN-DENORM-DAG: v_rcp_f32_e32 [[RCP2:v[0-9]+]], v{{[0-9]+}}
; GCN-DENORM-DAG: v_mul_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[RCP2]]

; GCN-DENORM-DAG: v_div_fmas_f32
; GCN-DENORM-DAG: v_div_fmas_f32
; GCN-DENORM-DAG: v_div_fixup_f32 {{.*}}, -2.0{{$}}
; GCN-DENORM-DAG: v_div_fixup_f32 {{.*}}, -2.0{{$}}

; GCN-FLUSH-DAG:  v_rcp_f32_e32
; GCN-FLUSH-DAG:  v_rcp_f32_e64

; GCN-NOT:        v_cmp_gt_f32_e64
; GCN-NOT:        v_cndmask_b32_e32
; GCN-FLUSH-NOT:  v_div

; GCN:            global_store_dwordx4
define amdgpu_kernel void @div_v4_c_by_minus_x_25ulp(<4 x float> addrspace(1)* %arg) {
  %load = load <4 x float>, <4 x float> addrspace(1)* %arg, align 16
  %neg = fneg <4 x float> %load
  %div = fdiv <4 x float> <float 2.000000e+00, float 1.000000e+00, float -1.000000e+00, float -2.000000e+00>, %neg, !fpmath !0
  store <4 x float> %div, <4 x float> addrspace(1)* %arg, align 16
  ret void
}

; GCN-LABEL: {{^}}div_v_by_x_25ulp:
; GCN-DAG:        s_load_dword [[VAL:s[0-9]+]], s[{{[0-9:]+}}], 0x0{{$}}

; GCN-DENORM-DAG: v_div_scale_f32
; GCN-DENORM-DAG: v_rcp_f32_e32
; GCN-DENORM-DAG: v_div_scale_f32
; GCN-DENORM:     v_div_fmas_f32
; GCN-DENORM:     v_div_fixup_f32 [[OUT:v[0-9]+]],

; GCN-FLUSH-DAG:  v_mov_b32_e32 [[L:v[0-9]+]], 0x6f800000
; GCN-FLUSH-DAG:  v_mov_b32_e32 [[S:v[0-9]+]], 0x2f800000
; GCN-FLUSH-DAG:  v_cmp_gt_f32_e64 vcc, |[[VAL]]|, [[L]]
; GCN-FLUSH-DAG:  v_cndmask_b32_e32 [[SCALE:v[0-9]+]], 1.0, [[S]], vcc
; GCN-FLUSH:      v_mul_f32_e32 [[PRESCALED:v[0-9]+]], [[VAL]], [[SCALE]]
; GCN-FLUSH:      v_rcp_f32_e32 [[RCP:v[0-9]+]], [[PRESCALED]]
; GCN-FLUSH:      v_mul_f32_e32 [[OUT:v[0-9]+]], [[SCALE]], [[RCP]]

; GCN:            global_store_dword v{{[0-9]+}}, [[OUT]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_v_by_x_25ulp(float addrspace(1)* %arg, float %num) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv float %num, %load, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_1_by_x_fast:
; GCN: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x0
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[VAL]]
; GCN: global_store_dword v{{[0-9]+}}, [[RCP]], s{{\[[0-9]:[0-9]+\]}}
define amdgpu_kernel void @div_1_by_x_fast(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv fast float 1.000000e+00, %load, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_minus_1_by_x_fast:
; GCN: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x0
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], -[[VAL]]
; GCN: global_store_dword v{{[0-9]+}}, [[RCP]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_minus_1_by_x_fast(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv fast float -1.000000e+00, %load, !fpmath !0
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_1_by_minus_x_fast:
; GCN: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x0
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], -[[VAL]]
; GCN: global_store_dword v{{[0-9]+}}, [[RCP]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_1_by_minus_x_fast(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %neg = fneg float %load, !fpmath !0
  %div = fdiv fast float 1.000000e+00, %neg
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_minus_1_by_minus_x_fast:
; GCN: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x0
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[VAL]]
; GCN: global_store_dword v{{[0-9]+}}, [[RCP]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @div_minus_1_by_minus_x_fast(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %neg = fsub float -0.000000e+00, %load, !fpmath !0
  %div = fdiv fast float -1.000000e+00, %neg
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_1_by_x_correctly_rounded:
; GCN-DAG: v_div_scale_f32
; GCN-DAG: v_rcp_f32_e32
; GCN-DAG: v_div_scale_f32
; GCN:     v_div_fmas_f32
; GCN:     v_div_fixup_f32
define amdgpu_kernel void @div_1_by_x_correctly_rounded(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv float 1.000000e+00, %load
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_minus_1_by_x_correctly_rounded:
; GCN-DAG: v_div_scale_f32
; GCN-DAG: v_rcp_f32_e32
; GCN-DAG: v_div_scale_f32
; GCN:     v_div_fmas_f32
; GCN:     v_div_fixup_f32
define amdgpu_kernel void @div_minus_1_by_x_correctly_rounded(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %div = fdiv float -1.000000e+00, %load
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_1_by_minus_x_correctly_rounded:
; GCN-DAG: v_div_scale_f32
; GCN-DAG: v_rcp_f32_e32
; GCN-DAG: v_div_scale_f32
; GCN:     v_div_fmas_f32
; GCN:     v_div_fixup_f32
define amdgpu_kernel void @div_1_by_minus_x_correctly_rounded(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %neg = fsub float -0.000000e+00, %load
  %div = fdiv float 1.000000e+00, %neg
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}div_minus_1_by_minus_x_correctly_rounded:
; GCN-DAG: v_div_scale_f32
; GCN-DAG: v_rcp_f32_e32
; GCN-DAG: v_div_scale_f32
; GCN:     v_div_fmas_f32
; GCN:     v_div_fixup_f32
define amdgpu_kernel void @div_minus_1_by_minus_x_correctly_rounded(float addrspace(1)* %arg) {
  %load = load float, float addrspace(1)* %arg, align 4
  %neg = fsub float -0.000000e+00, %load
  %div = fdiv float -1.000000e+00, %neg
  store float %div, float addrspace(1)* %arg, align 4
  ret void
}

!0 = !{float 2.500000e+00}
