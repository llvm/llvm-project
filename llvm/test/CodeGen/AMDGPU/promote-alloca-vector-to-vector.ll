; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -data-layout=A5 -mcpu=fiji -amdgpu-promote-alloca < %s | FileCheck -check-prefix=OPT %s

; GCN-LABEL: {{^}}float4_alloca_store4:
; OPT-LABEL: define amdgpu_kernel void @float4_alloca_store4

; GCN-NOT: buffer_
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32_e32 [[RES:v[0-9]+]], 4.0,
; GCN: store_dword v{{.+}}, [[RES]]

; OPT:  %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(5)* %alloca, i32 0, i32 %sel2
; OPT:  store <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, <4 x float> addrspace(5)* %alloca, align 4
; OPT:  %0 = load <4 x float>, <4 x float> addrspace(5)* %alloca
; OPT:  %1 = extractelement <4 x float> %0, i32 %sel2
; OPT:  store float %1, float addrspace(1)* %out, align 4

define amdgpu_kernel void @float4_alloca_store4(float addrspace(1)* %out, float addrspace(3)* %dummy_lds) {
entry:
  %alloca = alloca <4 x float>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(5)* %alloca, i32 0, i32 %sel2
  store <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, <4 x float> addrspace(5)* %alloca, align 4
  %load = load float, float addrspace(5)* %gep, align 4
  store float %load, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}float4_alloca_load4:
; OPT-LABEL: define amdgpu_kernel void @float4_alloca_load4

; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-NOT: v_cmp_
; GCN-NOT: v_cndmask_
; GCN:     v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     store_dwordx4 v{{.+}},

; OPT: %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(5)* %alloca, i32 0, i32 %sel2
; OPT: %0 = load <4 x float>, <4 x float> addrspace(5)* %alloca
; OPT: %1 = insertelement <4 x float> %0, float 1.000000e+00, i32 %sel2
; OPT: store <4 x float> %1, <4 x float> addrspace(5)* %alloca
; OPT: %load = load <4 x float>, <4 x float> addrspace(5)* %alloca, align 4
; OPT:  store <4 x float> %load, <4 x float> addrspace(1)* %out, align 4

define amdgpu_kernel void @float4_alloca_load4(<4 x float> addrspace(1)* %out, float addrspace(3)* %dummy_lds) {
entry:
  %alloca = alloca <4 x float>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(5)* %alloca, i32 0, i32 %sel2
  store float 1.0, float addrspace(5)* %gep, align 4
  %load = load <4 x float>, <4 x float> addrspace(5)* %alloca, align 4
  store <4 x float> %load, <4 x float> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}half4_alloca_store4:
; OPT-LABEL: define amdgpu_kernel void @half4_alloca_store4

; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0x44004200
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0x40003c00
; GCN:     v_lshrrev_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s{{\[}}[[SL]]:[[SH]]]

; OPT: %gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(5)* %alloca, i32 0, i32 %sel2
; OPT: store <4 x half> <half 0xH3C00, half 0xH4000, half 0xH4200, half 0xH4400>, <4 x half> addrspace(5)* %alloca, align 2
; OPT: %0 = load <4 x half>, <4 x half> addrspace(5)* %alloca
; OPT: %1 = extractelement <4 x half> %0, i32 %sel2
; OPT: store half %1, half addrspace(1)* %out, align 2

define amdgpu_kernel void @half4_alloca_store4(half addrspace(1)* %out, half addrspace(3)* %dummy_lds) {
entry:
  %alloca = alloca <4 x half>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(5)* %alloca, i32 0, i32 %sel2
  store <4 x half> <half 1.0, half 2.0, half 3.0, half 4.0>, <4 x half> addrspace(5)* %alloca, align 2
  %load = load half, half addrspace(5)* %gep, align 2
  store half %load, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}half4_alloca_load4:
; OPT-LABEL: define amdgpu_kernel void @half4_alloca_load4

; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0xffff

; OPT: %gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(5)* %alloca, i32 0, i32 %sel2
; OPT: %0 = load <4 x half>, <4 x half> addrspace(5)* %alloca
; OPT: %1 = insertelement <4 x half> %0, half 0xH3C00, i32 %sel2
; OPT: store <4 x half> %1, <4 x half> addrspace(5)* %alloca
; OPT: %load = load <4 x half>, <4 x half> addrspace(5)* %alloca, align 2
; OPT: store <4 x half> %load, <4 x half> addrspace(1)* %out, align 2

define amdgpu_kernel void @half4_alloca_load4(<4 x half> addrspace(1)* %out, half addrspace(3)* %dummy_lds) {
entry:
  %alloca = alloca <4 x half>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(5)* %alloca, i32 0, i32 %sel2
  store half 1.0, half addrspace(5)* %gep, align 4
  %load = load <4 x half>, <4 x half> addrspace(5)* %alloca, align 2
  store <4 x half> %load, <4 x half> addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}short4_alloca_store4:
; OPT-LABEL: define amdgpu_kernel void @short4_alloca_store4

; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0x40003
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0x20001
; GCN:     v_lshrrev_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s{{\[}}[[SL]]:[[SH]]]

; OPT: %gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(5)* %alloca, i32 0, i32 %sel2
; OPT: store <4 x i16> <i16 1, i16 2, i16 3, i16 4>, <4 x i16> addrspace(5)* %alloca, align 2
; OPT: %0 = load <4 x i16>, <4 x i16> addrspace(5)* %alloca
; OPT: %1 = extractelement <4 x i16> %0, i32 %sel2
; OPT: store i16 %1, i16 addrspace(1)* %out, align 2

define amdgpu_kernel void @short4_alloca_store4(i16 addrspace(1)* %out, i16 addrspace(3)* %dummy_lds) {
entry:
  %alloca = alloca <4 x i16>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(5)* %alloca, i32 0, i32 %sel2
  store <4 x i16> <i16 1, i16 2, i16 3, i16 4>, <4 x i16> addrspace(5)* %alloca, align 2
  %load = load i16, i16 addrspace(5)* %gep, align 2
  store i16 %load, i16 addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}short4_alloca_load4:
; OPT-LABEL: define amdgpu_kernel void @short4_alloca_load4

; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0xffff

; OPT: %gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(5)* %alloca, i32 0, i32 %sel2
; OPT: %0 = load <4 x i16>, <4 x i16> addrspace(5)* %alloca
; OPT: %1 = insertelement <4 x i16> %0, i16 1, i32 %sel2
; OPT: store <4 x i16> %1, <4 x i16> addrspace(5)* %alloca
; OPT: %load = load <4 x i16>, <4 x i16> addrspace(5)* %alloca, align 2
; OPT: store <4 x i16> %load, <4 x i16> addrspace(1)* %out, align 2

define amdgpu_kernel void @short4_alloca_load4(<4 x i16> addrspace(1)* %out, i16 addrspace(3)* %dummy_lds) {
entry:
  %alloca = alloca <4 x i16>, align 16, addrspace(5)
  %x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %y = tail call i32 @llvm.amdgcn.workitem.id.y()
  %c1 = icmp uge i32 %x, 3
  %c2 = icmp uge i32 %y, 3
  %sel1 = select i1 %c1, i32 1, i32 2
  %sel2 = select i1 %c2, i32 0, i32 %sel1
  %gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(5)* %alloca, i32 0, i32 %sel2
  store i16 1, i16 addrspace(5)* %gep, align 4
  %load = load <4 x i16>, <4 x i16> addrspace(5)* %alloca, align 2
  store <4 x i16> %load, <4 x i16> addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}ptr_alloca_bitcast:
; OPT-LABEL: define i64 @ptr_alloca_bitcast

; GCN-NOT: buffer_
; GCN: v_mov_b32_e32 v1, 0

; OPT: %private_iptr = alloca <2 x i32>, align 8, addrspace(5)
; OPT: %cast = bitcast <2 x i32> addrspace(5)* %private_iptr to i64 addrspace(5)*
; OPT: %tmp1 = load i64, i64 addrspace(5)* %cast, align 8

define i64 @ptr_alloca_bitcast() {
entry:
  %private_iptr = alloca <2 x i32>, align 8, addrspace(5)
  %cast = bitcast <2 x i32> addrspace(5)* %private_iptr to i64 addrspace(5)*
  %tmp1 = load i64, i64 addrspace(5)* %cast, align 8
  ret i64 %tmp1
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()
