;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s
;RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI %s

;EG: {{^}}test_select_v2i32:
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: {{^}}test_select_v2i32:
;SI: v_cndmask_b32_e64
;SI: v_cndmask_b32_e64

define void @test_select_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in0, <2 x i32> addrspace(1)* %in1) {
entry:
  %0 = load <2 x i32> addrspace(1)* %in0
  %1 = load <2 x i32> addrspace(1)* %in1
  %cmp = icmp ne <2 x i32> %0, %1
  %result = select <2 x i1> %cmp, <2 x i32> %0, <2 x i32> %1
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

;EG: {{^}}test_select_v2f32:
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: {{^}}test_select_v2f32:
;SI: v_cndmask_b32_e64
;SI: v_cndmask_b32_e64

define void @test_select_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in0, <2 x float> addrspace(1)* %in1) {
entry:
  %0 = load <2 x float> addrspace(1)* %in0
  %1 = load <2 x float> addrspace(1)* %in1
  %cmp = fcmp une <2 x float> %0, %1
  %result = select <2 x i1> %cmp, <2 x float> %0, <2 x float> %1
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

;EG: {{^}}test_select_v4i32:
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

;SI: {{^}}test_select_v4i32:
;SI: v_cndmask_b32_e64
;SI: v_cndmask_b32_e64
;SI: v_cndmask_b32_e64
;SI: v_cndmask_b32_e64

define void @test_select_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in0, <4 x i32> addrspace(1)* %in1) {
entry:
  %0 = load <4 x i32> addrspace(1)* %in0
  %1 = load <4 x i32> addrspace(1)* %in1
  %cmp = icmp ne <4 x i32> %0, %1
  %result = select <4 x i1> %cmp, <4 x i32> %0, <4 x i32> %1
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

;EG: {{^}}test_select_v4f32:
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;EG: CNDE_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @test_select_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in0, <4 x float> addrspace(1)* %in1) {
entry:
  %0 = load <4 x float> addrspace(1)* %in0
  %1 = load <4 x float> addrspace(1)* %in1
  %cmp = fcmp une <4 x float> %0, %1
  %result = select <4 x i1> %cmp, <4 x float> %0, <4 x float> %1
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
