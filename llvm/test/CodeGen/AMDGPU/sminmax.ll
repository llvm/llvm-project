; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}s_abs_i32:
; GCN: s_abs_i32
; GCN: s_add_i32

; EG: MAX_INT
define amdgpu_kernel void @s_abs_i32(i32 addrspace(1)* %out, i32 %val) nounwind {
  %neg = sub i32 0, %val
  %cond = icmp sgt i32 %val, %neg
  %res = select i1 %cond, i32 %val, i32 %neg
  %res2 = add i32 %res, 2
  store i32 %res2, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_abs_i32:
; GCN: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0, [[SRC:v[0-9]+]]
; GCN: v_max_i32_e32 {{v[0-9]+}}, [[NEG]], [[SRC]]
; GCN: v_add_i32

; EG: MAX_INT
define amdgpu_kernel void @v_abs_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %src) nounwind {
  %val = load i32, i32 addrspace(1)* %src, align 4
  %neg = sub i32 0, %val
  %cond = icmp sgt i32 %val, %neg
  %res = select i1 %cond, i32 %val, i32 %neg
  %res2 = add i32 %res, 2
  store i32 %res2, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_abs_i32_repeat_user:
; GCN: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0, [[SRC:v[0-9]+]]
; GCN: v_max_i32_e32 [[MAX:v[0-9]+]], [[NEG]], [[SRC]]
; GCN: v_mul_lo_i32 v{{[0-9]+}}, [[MAX]], [[MAX]]
define amdgpu_kernel void @v_abs_i32_repeat_user(i32 addrspace(1)* %out, i32 addrspace(1)* %src) nounwind {
  %val = load i32, i32 addrspace(1)* %src, align 4
  %neg = sub i32 0, %val
  %cond = icmp sgt i32 %val, %neg
  %res = select i1 %cond, i32 %val, i32 %neg
  %mul = mul i32 %res, %res
  store i32 %mul, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_abs_v2i32:
; GCN: s_abs_i32
; GCN: s_abs_i32
; GCN: s_add_i32
; GCN: s_add_i32

; EG: MAX_INT
; EG: MAX_INT
define amdgpu_kernel void @s_abs_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %val) nounwind {
  %z0 = insertelement <2 x i32> undef, i32 0, i32 0
  %z1 = insertelement <2 x i32> %z0, i32 0, i32 1
  %t0 = insertelement <2 x i32> undef, i32 2, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 2, i32 1
  %neg = sub <2 x i32> %z1, %val
  %cond = icmp sgt <2 x i32> %val, %neg
  %res = select <2 x i1> %cond, <2 x i32> %val, <2 x i32> %neg
  %res2 = add <2 x i32> %res, %t1
  store <2 x i32> %res2, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_abs_v2i32:
; GCN-DAG: v_sub_i32_e32 [[NEG0:v[0-9]+]], vcc, 0, [[SRC0:v[0-9]+]]
; GCN-DAG: v_sub_i32_e32 [[NEG1:v[0-9]+]], vcc, 0, [[SRC1:v[0-9]+]]

; GCN-DAG: v_max_i32_e32 {{v[0-9]+}}, [[NEG0]], [[SRC0]]
; GCN-DAG: v_max_i32_e32 {{v[0-9]+}}, [[NEG1]], [[SRC1]]

; GCN: v_add_i32
; GCN: v_add_i32

; EG: MAX_INT
; EG: MAX_INT
define amdgpu_kernel void @v_abs_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %src) nounwind {
  %z0 = insertelement <2 x i32> undef, i32 0, i32 0
  %z1 = insertelement <2 x i32> %z0, i32 0, i32 1
  %t0 = insertelement <2 x i32> undef, i32 2, i32 0
  %t1 = insertelement <2 x i32> %t0, i32 2, i32 1
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %src, align 4
  %neg = sub <2 x i32> %z1, %val
  %cond = icmp sgt <2 x i32> %val, %neg
  %res = select <2 x i1> %cond, <2 x i32> %val, <2 x i32> %neg
  %res2 = add <2 x i32> %res, %t1
  store <2 x i32> %res2, <2 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_abs_v4i32:
; TODO: this should use s_abs_i32
; GCN: s_abs_i32
; GCN: s_abs_i32
; GCN: s_abs_i32
; GCN: s_abs_i32

; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32
; GCN: s_add_i32

; EG: MAX_INT
; EG: MAX_INT
; EG: MAX_INT
; EG: MAX_INT
define amdgpu_kernel void @s_abs_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %val) nounwind {
  %z0 = insertelement <4 x i32> undef, i32 0, i32 0
  %z1 = insertelement <4 x i32> %z0, i32 0, i32 1
  %z2 = insertelement <4 x i32> %z1, i32 0, i32 2
  %z3 = insertelement <4 x i32> %z2, i32 0, i32 3
  %t0 = insertelement <4 x i32> undef, i32 2, i32 0
  %t1 = insertelement <4 x i32> %t0, i32 2, i32 1
  %t2 = insertelement <4 x i32> %t1, i32 2, i32 2
  %t3 = insertelement <4 x i32> %t2, i32 2, i32 3
  %neg = sub <4 x i32> %z3, %val
  %cond = icmp sgt <4 x i32> %val, %neg
  %res = select <4 x i1> %cond, <4 x i32> %val, <4 x i32> %neg
  %res2 = add <4 x i32> %res, %t3
  store <4 x i32> %res2, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_abs_v4i32:
; GCN-DAG: v_sub_i32_e32 [[NEG0:v[0-9]+]], vcc, 0, [[SRC0:v[0-9]+]]
; GCN-DAG: v_sub_i32_e32 [[NEG1:v[0-9]+]], vcc, 0, [[SRC1:v[0-9]+]]
; GCN-DAG: v_sub_i32_e32 [[NEG2:v[0-9]+]], vcc, 0, [[SRC2:v[0-9]+]]
; GCN-DAG: v_sub_i32_e32 [[NEG3:v[0-9]+]], vcc, 0, [[SRC3:v[0-9]+]]

; GCN-DAG: v_max_i32_e32 {{v[0-9]+}}, [[NEG0]], [[SRC0]]
; GCN-DAG: v_max_i32_e32 {{v[0-9]+}}, [[NEG1]], [[SRC1]]
; GCN-DAG: v_max_i32_e32 {{v[0-9]+}}, [[NEG2]], [[SRC2]]
; GCN-DAG: v_max_i32_e32 {{v[0-9]+}}, [[NEG3]], [[SRC3]]

; GCN: v_add_i32
; GCN: v_add_i32
; GCN: v_add_i32
; GCN: v_add_i32

; EG: MAX_INT
; EG: MAX_INT
; EG: MAX_INT
; EG: MAX_INT
define amdgpu_kernel void @v_abs_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %src) nounwind {
  %z0 = insertelement <4 x i32> undef, i32 0, i32 0
  %z1 = insertelement <4 x i32> %z0, i32 0, i32 1
  %z2 = insertelement <4 x i32> %z1, i32 0, i32 2
  %z3 = insertelement <4 x i32> %z2, i32 0, i32 3
  %t0 = insertelement <4 x i32> undef, i32 2, i32 0
  %t1 = insertelement <4 x i32> %t0, i32 2, i32 1
  %t2 = insertelement <4 x i32> %t1, i32 2, i32 2
  %t3 = insertelement <4 x i32> %t2, i32 2, i32 3
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %src, align 4
  %neg = sub <4 x i32> %z3, %val
  %cond = icmp sgt <4 x i32> %val, %neg
  %res = select <4 x i1> %cond, <4 x i32> %val, <4 x i32> %neg
  %res2 = add <4 x i32> %res, %t3
  store <4 x i32> %res2, <4 x i32> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_min_max_i32:
; GCN: s_load_dword [[VAL0:s[0-9]+]]
; GCN: s_load_dword [[VAL1:s[0-9]+]]

; GCN-DAG: s_min_i32 s{{[0-9]+}}, [[VAL0]], [[VAL1]]
; GCN-DAG: s_max_i32 s{{[0-9]+}}, [[VAL0]], [[VAL1]]
define amdgpu_kernel void @s_min_max_i32(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 %val0, i32 %val1) nounwind {
  %cond0 = icmp sgt i32 %val0, %val1
  %sel0 = select i1 %cond0, i32 %val0, i32 %val1
  %sel1 = select i1 %cond0, i32 %val1, i32 %val0

  store volatile i32 %sel0, i32 addrspace(1)* %out0, align 4
  store volatile i32 %sel1, i32 addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_min_max_i32:
; GCN: {{buffer|flat}}_load_dword [[VAL0:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[VAL1:v[0-9]+]]

; GCN-DAG: v_min_i32_e32 v{{[0-9]+}}, [[VAL1]], [[VAL0]]
; GCN-DAG: v_max_i32_e32 v{{[0-9]+}}, [[VAL1]], [[VAL0]]
define amdgpu_kernel void @v_min_max_i32(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 addrspace(1)* %ptr0, i32 addrspace(1)* %ptr1) nounwind {
  %val0 = load volatile i32, i32 addrspace(1)* %ptr0
  %val1 = load volatile i32, i32 addrspace(1)* %ptr1

  %cond0 = icmp sgt i32 %val0, %val1
  %sel0 = select i1 %cond0, i32 %val0, i32 %val1
  %sel1 = select i1 %cond0, i32 %val1, i32 %val0

  store volatile i32 %sel0, i32 addrspace(1)* %out0, align 4
  store volatile i32 %sel1, i32 addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_min_max_v4i32:
; GCN-DAG: s_min_i32
; GCN-DAG: s_min_i32
; GCN-DAG: s_min_i32
; GCN-DAG: s_min_i32
; GCN-DAG: s_max_i32
; GCN-DAG: s_max_i32
; GCN-DAG: s_max_i32
; GCN-DAG: s_max_i32
define amdgpu_kernel void @s_min_max_v4i32(<4 x i32> addrspace(1)* %out0, <4 x i32> addrspace(1)* %out1, <4 x i32> %val0, <4 x i32> %val1) nounwind {
  %cond0 = icmp sgt <4 x i32> %val0, %val1
  %sel0 = select <4 x i1> %cond0, <4 x i32> %val0, <4 x i32> %val1
  %sel1 = select <4 x i1> %cond0, <4 x i32> %val1, <4 x i32> %val0

  store volatile <4 x i32> %sel0, <4 x i32> addrspace(1)* %out0, align 4
  store volatile <4 x i32> %sel1, <4 x i32> addrspace(1)* %out1, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_min_max_i32_user:
; GCN: v_cmp_gt_i32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
define amdgpu_kernel void @v_min_max_i32_user(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 addrspace(1)* %ptr0, i32 addrspace(1)* %ptr1) nounwind {
  %val0 = load volatile i32, i32 addrspace(1)* %ptr0
  %val1 = load volatile i32, i32 addrspace(1)* %ptr1

  %cond0 = icmp sgt i32 %val0, %val1
  %sel0 = select i1 %cond0, i32 %val0, i32 %val1
  %sel1 = select i1 %cond0, i32 %val1, i32 %val0

  store volatile i32 %sel0, i32 addrspace(1)* %out0, align 4
  store volatile i32 %sel1, i32 addrspace(1)* %out1, align 4
  store volatile i1 %cond0, i1 addrspace(1)* undef
  ret void
}
