; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mtriple=amdgcn -mcpu=tahiti < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX89 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -verify-machineinstrs -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s

; Test expansion of scalar selects on vectors.
; Evergreen not enabled since it seems to be having problems with doubles.

; GCN-LABEL: {{^}}v_select_v2i8:
; SI: v_cndmask_b32
; SI-NOT: cndmask

; GFX9: v_cndmask_b32
; GFX9-NOT: cndmask

; This is worse when i16 is legal and packed is not because
; SelectionDAGBuilder for some reason changes the select type.
; VI: s_cselect_b64
; VI: v_cndmask_b32
define amdgpu_kernel void @v_select_v2i8(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <2 x i8>, ptr addrspace(1) %a.ptr, align 2
  %b = load <2 x i8>, ptr addrspace(1) %b.ptr, align 2
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i8> %a, <2 x i8> %b
  store <2 x i8> %select, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}v_select_v4i8:
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v4i8(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <4 x i8>, ptr addrspace(1) %a.ptr
  %b = load <4 x i8>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v8i8:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v8i8(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <8 x i8>, ptr addrspace(1) %a.ptr
  %b = load <8 x i8>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i8> %a, <8 x i8> %b
  store <8 x i8> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v16i8:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v16i8(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <16 x i8>, ptr addrspace(1) %a.ptr
  %b = load <16 x i8>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <16 x i8> %a, <16 x i8> %b
  store <16 x i8> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}select_v4i8:
; GFX89: s_cselect_b32
; GFX89-NOT: s_cselect_b32

; SI: s_cselect_b32
; SI-NOT: cndmask
define amdgpu_kernel void @select_v4i8(ptr addrspace(1) %out, <4 x i8> %a, <4 x i8> %b, i8 %c) #0 {
  %cmp = icmp eq i8 %c, 0
  %select = select i1 %cmp, <4 x i8> %a, <4 x i8> %b
  store <4 x i8> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}select_v2i16:
; GFX89: s_load_dwordx4
; GFX89: s_cselect_b32
; GFX89-NOT: s_cselect_b32

; SI: s_cselect_b32
; SI-NOT: v_cndmask_b32e
define amdgpu_kernel void @select_v2i16(ptr addrspace(1) %out, <2 x i16> %a, <2 x i16> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v2i16:
; GCN: buffer_load_dword v
; GCN: buffer_load_dword v
; GCN: v_cndmask_b32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <2 x i16>, ptr addrspace(1) %a.ptr
  %b = load <2 x i16>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v3i16:
; SI: v_cndmask_b32_e32
; SI: cndmask
; SI-NOT: cndmask

; VI: s_cselect_b64
; GFX9: cndmask
; GFX9: cndmask
define amdgpu_kernel void @v_select_v3i16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <3 x i16>, ptr addrspace(1) %a.ptr
  %b = load <3 x i16>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <3 x i16> %a, <3 x i16> %b
  store <3 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v4i16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <4 x i16>, ptr addrspace(1) %a.ptr
  %b = load <4 x i16>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i16> %a, <4 x i16> %b
  store <4 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v8i16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v8i16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <8 x i16>, ptr addrspace(1) %a.ptr
  %b = load <8 x i16>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i16> %a, <8 x i16> %b
  store <8 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v16i16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v16i16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <16 x i16>, ptr addrspace(1) %a.ptr
  %b = load <16 x i16>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <16 x i16> %a, <16 x i16> %b
  store <16 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v32i16:
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
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v32i16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <32 x i16>, ptr addrspace(1) %a.ptr
  %b = load <32 x i16>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <32 x i16> %a, <32 x i16> %b
  store <32 x i16> %select, ptr addrspace(1) %out, align 4
  ret void
}

; FIXME: Expansion with bitwise operations may be better if doing a
; vector select with SGPR inputs.

; GCN-LABEL: {{^}}s_select_v2i32:
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @s_select_v2i32(ptr addrspace(1) %out, <2 x i32> %a, <2 x i32> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x i32> %a, <2 x i32> %b
  store <2 x i32> %select, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}s_select_v4i32:
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @s_select_v4i32(ptr addrspace(1) %out, <4 x i32> %a, <4 x i32> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x i32> %a, <4 x i32> %b
  store <4 x i32> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}v_select_v4i32:
; GCN: buffer_load_dwordx4
; GCN: s_cmp_lt_u32 s{{[0-9]+}}, 32
; GCN: s_cselect_b64 vcc, -1, 0
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @v_select_v4i32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %cond) #0 {
bb:
  %tmp2 = icmp ult i32 %cond, 32
  %val = load <4 x i32>, ptr addrspace(1) %in
  %tmp3 = select i1 %tmp2, <4 x i32> %val, <4 x i32> zeroinitializer
  store <4 x i32> %tmp3, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v8i32:
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
define amdgpu_kernel void @select_v8i32(ptr addrspace(1) %out, <8 x i32> %a, <8 x i32> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x i32> %a, <8 x i32> %b
  store <8 x i32> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}s_select_v2f32:
; GCN-DAG: s_cmp_eq_u32 s{{[0-9]+}}, 0{{$}}
; GCN-DAG: s_cselect_b32
; GCN-DAG: s_cselect_b32
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @s_select_v2f32(ptr addrspace(1) %out, <2 x float> %a, <2 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x float> %a, <2 x float> %b
  store <2 x float> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}s_select_v3f32:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0{{$}}

; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32

; GCN: buffer_store_dwordx
define amdgpu_kernel void @s_select_v3f32(ptr addrspace(1) %out, <3 x float> %a, <3 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <3 x float> %a, <3 x float> %b
  store <3 x float> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}s_select_v4f32:
; GCN: s_load_dwordx8
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0{{$}}

; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32

; GCN: buffer_store_dwordx4
define amdgpu_kernel void @s_select_v4f32(ptr addrspace(1) %out, <4 x float> %a, <4 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x float> %a, <4 x float> %b
  store <4 x float> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}v_select_v4f32:
; GCN: buffer_load_dwordx4
; GCN: s_cmp_lt_u32 s{{[0-9]+}}, 32
; GCN: s_cselect_b64 vcc, -1, 0
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}, vcc
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @v_select_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %cond) #0 {
bb:
  %tmp2 = icmp ult i32 %cond, 32
  %val = load <4 x float>, ptr addrspace(1) %in
  %tmp3 = select i1 %tmp2, <4 x float> %val, <4 x float> zeroinitializer
  store <4 x float> %tmp3, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}s_select_v5f32:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0{{$}}

; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32

; GCN: buffer_store_dwordx
define amdgpu_kernel void @s_select_v5f32(ptr addrspace(1) %out, <5 x float> %a, <5 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <5 x float> %a, <5 x float> %b
  store <5 x float> %select, ptr addrspace(1) %out, align 16
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
define amdgpu_kernel void @select_v8f32(ptr addrspace(1) %out, <8 x float> %a, <8 x float> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x float> %a, <8 x float> %b
  store <8 x float> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v2f64:
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
define amdgpu_kernel void @select_v2f64(ptr addrspace(1) %out, <2 x double> %a, <2 x double> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x double> %a, <2 x double> %b
  store <2 x double> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v4f64:
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
define amdgpu_kernel void @select_v4f64(ptr addrspace(1) %out, <4 x double> %a, <4 x double> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x double> %a, <4 x double> %b
  store <4 x double> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}select_v8f64:
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
; GCN: s_cselect_b32
define amdgpu_kernel void @select_v8f64(ptr addrspace(1) %out, <8 x double> %a, <8 x double> %b, i32 %c) #0 {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <8 x double> %a, <8 x double> %b
  store <8 x double> %select, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}v_select_v2f16:
; GCN: v_cndmask_b32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <2 x half>, ptr addrspace(1) %a.ptr
  %b = load <2 x half>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <2 x half> %a, <2 x half> %b
  store <2 x half> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v3f16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v3f16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <3 x half>, ptr addrspace(1) %a.ptr
  %b = load <3 x half>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <3 x half> %a, <3 x half> %b
  store <3 x half> %select, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_select_v4f16:
; GCN: v_cndmask_b32_e32
; GCN: v_cndmask_b32_e32
; GCN-NOT: cndmask
define amdgpu_kernel void @v_select_v4f16(ptr addrspace(1) %out, ptr addrspace(1) %a.ptr, ptr addrspace(1) %b.ptr, i32 %c) #0 {
  %a = load <4 x half>, ptr addrspace(1) %a.ptr
  %b = load <4 x half>, ptr addrspace(1) %b.ptr
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, <4 x half> %a, <4 x half> %b
  store <4 x half> %select, ptr addrspace(1) %out, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
