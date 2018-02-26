; RUN:  llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s

; Test expansion of scalar selects on vectors.
; Evergreen not enabled since it seems to be having problems with doubles.

; GCN-LABEL: {{^}}v_select_v2i8:
; SI: v_cndmask_b32
; SI-NOT: cndmask

; GFX9: v_cndmask_b32
; GFX9-NOT: cndmask

; This is worse when i16 is legal and packed is not because
; SelectionDAGBuilder for some reason changes the select type.
; VI: v_cndmask_b32
; VI: v_cndmask_b32
define amdgpu_kernel void @v_select_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %a.ptr, <2 x i8> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <2 x i8>, <2 x i8> addrspace(1)* %a.ptr, align 2
  %b = load <2 x i8>, <2 x i8> addrspace(1)* %b.ptr, align 2
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i8> %a, <2 x i8> %b
  store <2 x i8> %select, <2 x i8> addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}v_select_v4i8:
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> addrspace(1)* %a.ptr, <4 x i8> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <4 x i8>, <4 x i8> addrspace(1)* %a.ptr
  %b = load <4 x i8>, <4 x i8> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %select, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v8i8:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v8i8(<8 x i8> addrspace(1)* %out, <8 x i8> addrspace(1)* %a.ptr, <8 x i8> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <8 x i8>, <8 x i8> addrspace(1)* %a.ptr
  %b = load <8 x i8>, <8 x i8> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i8> %a, <8 x i8> %b
  store <8 x i8> %select, <8 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v16i8:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v16i8(<16 x i8> addrspace(1)* %out, <16 x i8> addrspace(1)* %a.ptr, <16 x i8> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <16 x i8>, <16 x i8> addrspace(1)* %a.ptr
  %b = load <16 x i8>, <16 x i8> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <16 x i8> %a, <16 x i8> %b
  store <16 x i8> %select, <16 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}select_v4i8:
; GCN: v_cndmask_b32
; GCN-NOT: cndmask
define amdgpu_kernel void @select_v4i8(<4 x i8> addrspace(1)* %out, <4 x i8> %a, <4 x i8> %b, i8 %c) #0 {
  %cmp = icmp eq i8 %c, 0
  %select = select i1 %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %select, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}select_v2i16:
; GCN: v_cndmask_b32_e32
; GCN-NOT: v_cndmask_b32
define amdgpu_kernel void @select_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> %a, <2 x i16> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %select, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v2i16:
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %a.ptr, <2 x i16> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %a.ptr
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %select, <2 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v3i16:
; SI: v_cndmask_b32_e32
; SI: cndmask
; SI-NOT: cndmask

; GFX9: v_cndmask_b32_e32
; GFX9: cndmask
; GFX9-NOT: cndmask

; VI: v_cndmask_b32
; VI: v_cndmask_b32
; VI: v_cndmask_b32
define amdgpu_kernel void @v_select_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> addrspace(1)* %a.ptr, <3 x i16> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <3 x i16>, <3 x i16> addrspace(1)* %a.ptr
  %b = load <3 x i16>, <3 x i16> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <3 x i16> %a, <3 x i16> %b
  store <3 x i16> %select, <3 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v4i16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %a.ptr, <4 x i16> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %a.ptr
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %select, <4 x i16> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v8i16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v8i16(<8 x i16> addrspace(1)* %out, <8 x i16> addrspace(1)* %a.ptr, <8 x i16> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <8 x i16>, <8 x i16> addrspace(1)* %a.ptr
  %b = load <8 x i16>, <8 x i16> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i16> %a, <8 x i16> %b
  store <8 x i16> %select, <8 x i16> addrspace(1)* %out, align 4
  ret void
}

; FIXME: Expansion with bitwise operations may be better if doing a
; vector select with SGPR inputs.

; GCN-LABEL: {{^}}s_select_v2i32:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @s_select_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, <2 x i32> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i32> %a, <2 x i32> %b
  store <2 x i32> %select, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}s_select_v4i32:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @s_select_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %select, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}v_select_v4i32:
; GCN: buffer_load_dwordx4
; GCN: v_cmp_lt_u32_e64 vcc, s{{[0-9]+}}, 32
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @v_select_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %cond) #0 {
bb:
  %tmp2 = icmp ult i32 %cond, 32
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %in
  %tmp3 = select i1 %tmp2, <4 x i32> %val, <4 x i32> zeroinitializer
  store <4 x i32> %tmp3, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v8i32:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @select_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, <8 x i32> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i32> %a, <8 x i32> %b
  store <8 x i32> %select, <8 x i32> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}s_select_v2f32:
; GCN-DAG: s_load_dwordx2 s{{\[}}[[ALO:[0-9]+]]:[[AHI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: s_load_dwordx2 s{{\[}}[[BLO:[0-9]+]]:[[BHI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xd|0x34}}

; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[AHI]]
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[BHI]]
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[ALO]]
; GCN-DAG: v_cmp_eq_u32_e64 vcc, s{{[0-9]+}}, 0{{$}}

; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s[[BLO]]
; GCN-DAG: v_cndmask_b32_e32
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @s_select_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x float> %a, <2 x float> %b
  store <2 x float> %select, <2 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}s_select_v4f32:
; GCN: s_load_dwordx4
; GCN: s_load_dwordx4
; GCN: v_cmp_eq_u32_e64 vcc, s{{[0-9]+}}, 0{{$}}

; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32

; GCN: buffer_store_dwordx4
define amdgpu_kernel void @s_select_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, <4 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x float> %a, <4 x float> %b
  store <4 x float> %select, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}v_select_v4f32:
; GCN: buffer_load_dwordx4
; GCN: v_cmp_lt_u32_e64 vcc, s{{[0-9]+}}, 32
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @v_select_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in, i32 %cond) #0 {
bb:
  %tmp2 = icmp ult i32 %cond, 32
  %val = load <4 x float>, <4 x float> addrspace(1)* %in
  %tmp3 = select i1 %tmp2, <4 x float> %val, <4 x float> zeroinitializer
  store <4 x float> %tmp3, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v8f32:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @select_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %a, <8 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x float> %a, <8 x float> %b
  store <8 x float> %select, <8 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v2f64:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @select_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x double> %a, <2 x double> %b
  store <2 x double> %select, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v4f64:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @select_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x double> %a, <4 x double> %b
  store <4 x double> %select, <4 x double> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v8f64:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
define amdgpu_kernel void @select_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %a, <8 x double> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x double> %a, <8 x double> %b
  store <8 x double> %select, <8 x double> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}v_select_v2f16:
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %a.ptr, <2 x half> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <2 x half>, <2 x half> addrspace(1)* %a.ptr
  %b = load <2 x half>, <2 x half> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x half> %a, <2 x half> %b
  store <2 x half> %select, <2 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v3f16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v3f16(<3 x half> addrspace(1)* %out, <3 x half> addrspace(1)* %a.ptr, <3 x half> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <3 x half>, <3 x half> addrspace(1)* %a.ptr
  %b = load <3 x half>, <3 x half> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <3 x half> %a, <3 x half> %b
  store <3 x half> %select, <3 x half> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v4f16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v4f16(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %a.ptr, <4 x half> addrspace(1)* %b.ptr, i32 %c) #0 {
  %a = load <4 x half>, <4 x half> addrspace(1)* %a.ptr
  %b = load <4 x half>, <4 x half> addrspace(1)* %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x half> %a, <4 x half> %b
  store <4 x half> %select, <4 x half> addrspace(1)* %out, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
