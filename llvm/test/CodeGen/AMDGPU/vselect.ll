;RUN: llc < %s -mtriple=amdgcn -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=FUNC %s
;RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck --check-prefix=VI --check-prefix=FUNC %s
;RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck --check-prefix=EG --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_select_v2i32:

; EG-DAG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW]}}, KC0[3].Z
; EG-DAG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW]}}, KC0[3].Y

; VI: s_cmp_gt_i32
; VI: s_cselect_b32
; VI: s_cmp_gt_i32
; VI: s_cselect_b32

; SI-DAG: s_cmp_gt_i32
; SI-DAG: s_cselect_b32
; SI-DAG: s_cmp_gt_i32
; SI-DAG: s_cselect_b32

define amdgpu_kernel void @test_select_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1, <2 x i32> %val) {
entry:
  %load0 = load <2 x i32>, ptr addrspace(1) %in0
  %load1 = load <2 x i32>, ptr addrspace(1) %in1
  %cmp = icmp sgt <2 x i32> %load0, %load1
  %result = select <2 x i1> %cmp, <2 x i32> %val, <2 x i32> %load0
  store <2 x i32> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_select_v2f32:

; EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: v_cmp_neq_f32_e32 vcc
; SI: v_cndmask_b32_e32
; SI: v_cmp_neq_f32_e32 vcc
; SI: v_cndmask_b32_e32

define amdgpu_kernel void @test_select_v2f32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
entry:
  %0 = load <2 x float>, ptr addrspace(1) %in0
  %1 = load <2 x float>, ptr addrspace(1) %in1
  %cmp = fcmp une <2 x float> %0, %1
  %result = select <2 x i1> %cmp, <2 x float> %0, <2 x float> %1
  store <2 x float> %result, ptr addrspace(1) %out
  ret void
}

;FUNC-LABEL: {{^}}test_select_v4i32:

; EG-DAG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW]}}, KC0[4].X
; EG-DAG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, KC0[3].W
; EG-DAG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW]}}, KC0[3].Z
; EG-DAG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW]}}, KC0[3].Y

; VI: s_cselect_b32
; VI: s_cselect_b32
; VI: s_cselect_b32
; VI: s_cselect_b32

; SI-DAG: s_cselect_b32
; SI-DAG: s_cselect_b32
; SI-DAG: s_cselect_b32
; SI-DAG: s_cselect_b32

define amdgpu_kernel void @test_select_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1, <4 x i32> %val) {
entry:
  %load0 = load <4 x i32>, ptr addrspace(1) %in0
  %load1 = load <4 x i32>, ptr addrspace(1) %in1
  %cmp = icmp sgt <4 x i32> %load0, %load1
  %result = select <4 x i1> %cmp, <4 x i32> %val, <4 x i32> %load0
  store <4 x i32> %result, ptr addrspace(1) %out
  ret void
}

;FUNC-LABEL: {{^}}test_select_v4f32:
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
; SI: v_cndmask_b32_e32
define amdgpu_kernel void @test_select_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %in0, ptr addrspace(1) %in1) {
entry:
  %0 = load <4 x float>, ptr addrspace(1) %in0
  %1 = load <4 x float>, ptr addrspace(1) %in1
  %cmp = fcmp une <4 x float> %0, %1
  %result = select <4 x i1> %cmp, <4 x float> %0, <4 x float> %1
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}
