; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX9,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI,CIVI,GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=CI,CIVI,GCN %s

; GCN-LABEL: {{^}}s_abs_v2i16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_sub_i16 [[SUB:v[0-9]+]], 0, [[VAL]]
; GFX9: v_pk_max_i16 [[MAX:v[0-9]+]], [[VAL]], [[SUB]]
; GFX9: v_pk_add_u16 [[ADD:v[0-9]+]], [[MAX]], 2 op_sel_hi:[1,0]

; CIVI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; CIVI: s_sub_i32
; CIVI: s_sub_i32
; CIVI: s_max_i32
; CIVI: s_max_i32
; CIVI: s_add_i32
; CIVI-DAG: s_add_i32
; CIVI-DAG: s_and_b32
; CIVI-DAG: s_or_b32
define amdgpu_kernel void @s_abs_v2i16(ptr addrspace(1) %out, <2 x i16> %val) #0 {
  %neg = sub <2 x i16> zeroinitializer, %val
  %cond = icmp sgt <2 x i16> %val, %neg
  %res = select <2 x i1> %cond, <2 x i16> %val, <2 x i16> %neg
  %res2 = add <2 x i16> %res, <i16 2, i16 2>
  store <2 x i16> %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_abs_v2i16:
; GFX9: global_load_dword [[VAL:v[0-9]+]]
; GFX9: v_pk_sub_i16 [[SUB:v[0-9]+]], 0, [[VAL]]
; GFX9: v_pk_max_i16 [[MAX:v[0-9]+]], [[VAL]], [[SUB]]
; GFX9: v_pk_add_u16 [[ADD:v[0-9]+]], [[MAX]], 2 op_sel_hi:[1,0]

; VI-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; VI-DAG: v_lshrrev_b32_e32 v{{[0-9]+}}, 16,
; VI-DAG: v_sub_u16_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; VI-DAG: v_sub_u16_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; VI-DAG: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI-DAG: v_max_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_add_u16_e32 v{{[0-9]+}}, 2, v{{[0-9]+}}
; VI: v_add_u16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[TWO]]  dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-NOT: v_and_b32
; VI: v_or_b32_e32

; CI: buffer_load_dword v
; CI: v_lshrrev_b32_e32
; CI-DAG: v_sub_i32_e32
; CI-DAG: v_bfe_i32
; CI-DAG: v_bfe_i32
; CI-DAG: v_max_i32
; CI-DAG: v_max_i32
; CI-DAG: v_add_i32
; CI-DAG: v_add_i32
; CI-DAG: v_or_b32
define amdgpu_kernel void @v_abs_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %src) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x i16>, ptr addrspace(1) %src, i32 %tid
  %gep.out = getelementptr inbounds <2 x i16>, ptr addrspace(1) %out, i32 %tid
  %val = load <2 x i16>, ptr addrspace(1) %gep.in, align 4
  %neg = sub <2 x i16> zeroinitializer, %val
  %cond = icmp sgt <2 x i16> %val, %neg
  %res = select <2 x i1> %cond, <2 x i16> %val, <2 x i16> %neg
  %res2 = add <2 x i16> %res, <i16 2, i16 2>
  store <2 x i16> %res2, ptr addrspace(1) %gep.out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_abs_v2i16_2:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_sub_i16 [[SUB:v[0-9]+]], 0, [[VAL]]
; GFX9: v_pk_max_i16 [[MAX:v[0-9]+]], [[VAL]], [[SUB]]
; GFX9: v_pk_add_u16 [[ADD:v[0-9]+]], [[MAX]], 2 op_sel_hi:[1,0]
define amdgpu_kernel void @s_abs_v2i16_2(ptr addrspace(1) %out, <2 x i16> %val) #0 {
  %z0 = insertelement <2 x i16> undef, i16 0, i16 0
  %z1 = insertelement <2 x i16> %z0, i16 0, i16 1
  %t0 = insertelement <2 x i16> undef, i16 2, i16 0
  %t1 = insertelement <2 x i16> %t0, i16 2, i16 1
  %neg = sub <2 x i16> %z1, %val
  %cond = icmp sgt <2 x i16> %val, %neg
  %res = select <2 x i1> %cond, <2 x i16> %val, <2 x i16> %neg
  %res2 = add <2 x i16> %res, %t1
  store <2 x i16> %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_abs_v2i16_2:
; GFX9: global_load_dword [[VAL:v[0-9]+]]
; GFX9: v_pk_sub_i16 [[SUB:v[0-9]+]], 0, [[VAL]]
; GFX9: v_pk_max_i16 [[MAX:v[0-9]+]], [[VAL]], [[SUB]]
; GFX9: v_pk_add_u16 [[ADD:v[0-9]+]], [[MAX]], 2 op_sel_hi:[1,0]
define amdgpu_kernel void @v_abs_v2i16_2(ptr addrspace(1) %out, ptr addrspace(1) %src) #0 {
  %z0 = insertelement <2 x i16> undef, i16 0, i16 0
  %z1 = insertelement <2 x i16> %z0, i16 0, i16 1
  %t0 = insertelement <2 x i16> undef, i16 2, i16 0
  %t1 = insertelement <2 x i16> %t0, i16 2, i16 1
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <2 x i16>, ptr addrspace(1) %src, i32 %tid
  %val = load <2 x i16>, ptr addrspace(1) %gep.in, align 4
  %neg = sub <2 x i16> %z1, %val
  %cond = icmp sgt <2 x i16> %val, %neg
  %res = select <2 x i1> %cond, <2 x i16> %val, <2 x i16> %neg
  %res2 = add <2 x i16> %res, %t1
  store <2 x i16> %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_abs_v4i16:
; GFX9: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s[0:1], 0x24
; GFX9-DAG: v_pk_sub_i16 [[SUB0:v[0-9]+]], 0, s[[#LOAD + 2]]
; GFX9-DAG: v_pk_sub_i16 [[SUB1:v[0-9]+]], 0, s[[#LOAD + 3]]
; GFX9-DAG: v_pk_max_i16 [[MAX0:v[0-9]+]], s[[#LOAD + 2]], [[SUB0]]
; GFX9-DAG: v_pk_max_i16 [[MAX1:v[0-9]+]], s[[#LOAD + 3]], [[SUB1]]
; GFX9-DAG: v_pk_add_u16 [[ADD0:v[0-9]+]], [[MAX0]], 2 op_sel_hi:[1,0]
; GFX9-DAG: v_pk_add_u16 [[ADD1:v[0-9]+]], [[MAX1]], 2 op_sel_hi:[1,0]
define amdgpu_kernel void @s_abs_v4i16(ptr addrspace(1) %out, <4 x i16> %val) #0 {
  %z0 = insertelement <4 x i16> undef, i16 0, i16 0
  %z1 = insertelement <4 x i16> %z0, i16 0, i16 1
  %z2 = insertelement <4 x i16> %z1, i16 0, i16 2
  %z3 = insertelement <4 x i16> %z2, i16 0, i16 3
  %t0 = insertelement <4 x i16> undef, i16 2, i16 0
  %t1 = insertelement <4 x i16> %t0, i16 2, i16 1
  %t2 = insertelement <4 x i16> %t1, i16 2, i16 2
  %t3 = insertelement <4 x i16> %t2, i16 2, i16 3
  %neg = sub <4 x i16> %z3, %val
  %cond = icmp sgt <4 x i16> %val, %neg
  %res = select <4 x i1> %cond, <4 x i16> %val, <4 x i16> %neg
  %res2 = add <4 x i16> %res, %t3
  store <4 x i16> %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_abs_v4i16:
; GFX9: global_load_dwordx2 v[[[VAL0:[0-9]+]]:[[VAL1:[0-9]+]]]

; GFX9-DAG: v_pk_sub_i16 [[SUB0:v[0-9]+]], 0, v[[VAL0]]
; GFX9-DAG: v_pk_max_i16 [[MAX0:v[0-9]+]], v[[VAL0]], [[SUB0]]
; GFX9-DAG: v_pk_add_u16 [[ADD0:v[0-9]+]], [[MAX0]], 2 op_sel_hi:[1,0]

; GFX9-DAG: v_pk_sub_i16 [[SUB1:v[0-9]+]], 0, v[[VAL1]]
; GFX9-DAG: v_pk_max_i16 [[MAX1:v[0-9]+]], v[[VAL1]], [[SUB1]]
; GFX9-DAG: v_pk_add_u16 [[ADD1:v[0-9]+]], [[MAX1]], 2 op_sel_hi:[1,0]
define amdgpu_kernel void @v_abs_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %src) #0 {
  %z0 = insertelement <4 x i16> undef, i16 0, i16 0
  %z1 = insertelement <4 x i16> %z0, i16 0, i16 1
  %z2 = insertelement <4 x i16> %z1, i16 0, i16 2
  %z3 = insertelement <4 x i16> %z2, i16 0, i16 3
  %t0 = insertelement <4 x i16> undef, i16 2, i16 0
  %t1 = insertelement <4 x i16> %t0, i16 2, i16 1
  %t2 = insertelement <4 x i16> %t1, i16 2, i16 2
  %t3 = insertelement <4 x i16> %t2, i16 2, i16 3
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.in = getelementptr inbounds <4 x i16>, ptr addrspace(1) %src, i32 %tid
  %val = load <4 x i16>, ptr addrspace(1) %gep.in, align 4
  %neg = sub <4 x i16> %z3, %val
  %cond = icmp sgt <4 x i16> %val, %neg
  %res = select <4 x i1> %cond, <4 x i16> %val, <4 x i16> %neg
  %res2 = add <4 x i16> %res, %t3
  store <4 x i16> %res2, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_min_max_v2i16:
; GFX9: v_pk_max_i16
; GFX9: v_pk_min_i16
define amdgpu_kernel void @s_min_max_v2i16(ptr addrspace(1) %out0, ptr addrspace(1) %out1, <2 x i16> %val0, <2 x i16> %val1) #0 {
  %cond0 = icmp sgt <2 x i16> %val0, %val1
  %sel0 = select <2 x i1> %cond0, <2 x i16> %val0, <2 x i16> %val1
  %sel1 = select <2 x i1> %cond0, <2 x i16> %val1, <2 x i16> %val0

  store volatile <2 x i16> %sel0, ptr addrspace(1) %out0, align 4
  store volatile <2 x i16> %sel1, ptr addrspace(1) %out1, align 4
  ret void
}

; GCN-LABEL: {{^}}v_min_max_v2i16:
; GFX9: v_pk_max_i16
; GFX9: v_pk_min_i16
define amdgpu_kernel void @v_min_max_v2i16(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %ptr0, ptr addrspace(1) %ptr1) #0 {
  %val0 = load volatile <2 x i16>, ptr addrspace(1) %ptr0
  %val1 = load volatile <2 x i16>, ptr addrspace(1) %ptr1

  %cond0 = icmp sgt <2 x i16> %val0, %val1
  %sel0 = select <2 x i1> %cond0, <2 x i16> %val0, <2 x i16> %val1
  %sel1 = select <2 x i1> %cond0, <2 x i16> %val1, <2 x i16> %val0

  store volatile <2 x i16> %sel0, ptr addrspace(1) %out0, align 4
  store volatile <2 x i16> %sel1, ptr addrspace(1) %out1, align 4
  ret void
}

; GCN-LABEL: {{^}}s_min_max_v4i16:
; GFX9-DAG: v_pk_max_i16
; GFX9-DAG: v_pk_min_i16
; GFX9-DAG: v_pk_max_i16
; GFX9-DAG: v_pk_min_i16
define amdgpu_kernel void @s_min_max_v4i16(ptr addrspace(1) %out0, ptr addrspace(1) %out1, <4 x i16> %val0, <4 x i16> %val1) #0 {
  %cond0 = icmp sgt <4 x i16> %val0, %val1
  %sel0 = select <4 x i1> %cond0, <4 x i16> %val0, <4 x i16> %val1
  %sel1 = select <4 x i1> %cond0, <4 x i16> %val1, <4 x i16> %val0

  store volatile <4 x i16> %sel0, ptr addrspace(1) %out0, align 4
  store volatile <4 x i16> %sel1, ptr addrspace(1) %out1, align 4
  ret void
}

; GCN-LABEL: {{^}}v_min_max_v2i16_user:
define amdgpu_kernel void @v_min_max_v2i16_user(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %ptr0, ptr addrspace(1) %ptr1) #0 {
  %val0 = load volatile <2 x i16>, ptr addrspace(1) %ptr0
  %val1 = load volatile <2 x i16>, ptr addrspace(1) %ptr1

  %cond0 = icmp sgt <2 x i16> %val0, %val1
  %sel0 = select <2 x i1> %cond0, <2 x i16> %val0, <2 x i16> %val1
  %sel1 = select <2 x i1> %cond0, <2 x i16> %val1, <2 x i16> %val0

  store volatile <2 x i16> %sel0, ptr addrspace(1) %out0, align 4
  store volatile <2 x i16> %sel1, ptr addrspace(1) %out1, align 4
  store volatile <2 x i1> %cond0, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}u_min_max_v2i16:
; GFX9: v_pk_max_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_pk_min_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @u_min_max_v2i16(ptr addrspace(1) %out0, ptr addrspace(1) %out1, <2 x i16> %val0, <2 x i16> %val1) nounwind {
  %cond0 = icmp ugt <2 x i16> %val0, %val1
  %sel0 = select <2 x i1> %cond0, <2 x i16> %val0, <2 x i16> %val1
  %sel1 = select <2 x i1> %cond0, <2 x i16> %val1, <2 x i16> %val0

  store volatile <2 x i16> %sel0, ptr addrspace(1) %out0, align 4
  store volatile <2 x i16> %sel1, ptr addrspace(1) %out1, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
