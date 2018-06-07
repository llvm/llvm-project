; RUN: llc -amdgpu-codegenprepare-widen-constant-loads=0 -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -amdgpu-codegenprepare-widen-constant-loads=0 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}widen_i16_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_addk_i32 [[VAL]], 0x3e7
; GCN: s_or_b32 [[OR:s[0-9]+]], [[VAL]], 4
define amdgpu_kernel void @widen_i16_constant_load(i16 addrspace(4)* %arg) {
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %add = add i16 %load, 999
  %or = or i16 %add, 4
  store i16 %or, i16 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i16_constant_load_zext_i32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 [[TRUNC:s[0-9]+]], [[VAL]], 0xffff{{$}}
; GCN: s_addk_i32 [[TRUNC]], 0x3e7
; GCN: s_or_b32 [[OR:s[0-9]+]], [[TRUNC]], 4
define amdgpu_kernel void @widen_i16_constant_load_zext_i32(i16 addrspace(4)* %arg) {
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %ext = zext i16 %load to i32
  %add = add i32 %ext, 999
  %or = or i32 %add, 4
  store i32 %or, i32 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i16_constant_load_sext_i32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_sext_i32_i16 [[EXT:s[0-9]+]], [[VAL]]
; GCN: s_addk_i32 [[EXT]], 0x3e7
; GCN: s_or_b32 [[OR:s[0-9]+]], [[EXT]], 4
define amdgpu_kernel void @widen_i16_constant_load_sext_i32(i16 addrspace(4)* %arg) {
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %ext = sext i16 %load to i32
  %add = add i32 %ext, 999
  %or = or i32 %add, 4
  store i32 %or, i32 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i17_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_add_i32 [[ADD:s[0-9]+]], [[VAL]], 34
; GCN: s_or_b32 [[OR:s[0-9]+]], [[ADD]], 4
; GCN: s_bfe_u32 s{{[0-9]+}}, [[OR]], 0x10010
define amdgpu_kernel void @widen_i17_constant_load(i17 addrspace(4)* %arg) {
  %load = load i17, i17 addrspace(4)* %arg, align 4
  %add = add i17 %load, 34
  %or = or i17 %add, 4
  store i17 %or, i17 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_f16_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; SI: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], [[VAL]]
; SI: v_add_f32_e32 [[ADD:v[0-9]+]], 4.0, [[CVT]]

; VI: v_add_f16_e64 [[ADD:v[0-9]+]], [[VAL]], 4.0
define amdgpu_kernel void @widen_f16_constant_load(half addrspace(4)* %arg) {
  %load = load half, half addrspace(4)* %arg, align 4
  %add = fadd half %load, 4.0
  store half %add, half addrspace(1)* null
  ret void
}

; FIXME: valu usage on VI
; GCN-LABEL: {{^}}widen_v2i8_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]

; SI: s_add_i32
; SI: s_or_b32
; SI: s_addk_i32
; SI: s_and_b32
; SI: s_or_b32
; SI: s_or_b32

; VI: s_add_i32
; VI: v_add_u32_sdwa
; VI: v_or_b32_sdwa
; VI: v_or_b32_e32
define amdgpu_kernel void @widen_v2i8_constant_load(<2 x i8> addrspace(4)* %arg) {
  %load = load <2 x i8>, <2 x i8> addrspace(4)* %arg, align 4
  %add = add <2 x i8> %load, <i8 12, i8 44>
  %or = or <2 x i8> %add, <i8 4, i8 3>
  store <2 x i8> %or, <2 x i8> addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}no_widen_i16_constant_divergent_load:
; GCN: {{buffer|flat}}_load_ushort
define amdgpu_kernel void @no_widen_i16_constant_divergent_load(i16 addrspace(4)* %arg) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = zext i32 %tid to i64
  %gep.arg = getelementptr inbounds i16, i16 addrspace(4)* %arg, i64 %tid.ext
  %load = load i16, i16 addrspace(4)* %gep.arg, align 4
  %add = add i16 %load, 999
  %or = or i16 %add, 4
  store i16 %or, i16 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i1_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 {{s[0-9]+}}, [[VAL]], 1{{$}}
define amdgpu_kernel void @widen_i1_constant_load(i1 addrspace(4)* %arg) {
  %load = load i1, i1 addrspace(4)* %arg, align 4
  %and = and i1 %load, true
  store i1 %and, i1 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i16_zextload_i64_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 [[TRUNC:s[0-9]+]], [[VAL]], 0xffff{{$}}
; GCN: s_addk_i32 [[TRUNC]], 0x3e7
; GCN: s_or_b32 [[OR:s[0-9]+]], [[TRUNC]], 4
define amdgpu_kernel void @widen_i16_zextload_i64_constant_load(i16 addrspace(4)* %arg) {
  %load = load i16, i16 addrspace(4)* %arg, align 4
  %zext = zext i16 %load to i32
  %add = add i32 %zext, 999
  %or = or i32 %add, 4
  store i32 %or, i32 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i1_zext_to_i64_constant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_and_b32 [[AND:s[0-9]+]], [[VAL]], 1
; GCN: s_add_u32 [[ADD:s[0-9]+]], [[AND]], 0x3e7
; GCN: s_addc_u32 s{{[0-9]+}}, 0, 0
define amdgpu_kernel void @widen_i1_zext_to_i64_constant_load(i1 addrspace(4)* %arg) {
  %load = load i1, i1 addrspace(4)* %arg, align 4
  %zext = zext i1 %load to i64
  %add = add i64 %zext, 999
  store i64 %add, i64 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i16_constant32_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_addk_i32 [[VAL]], 0x3e7
; GCN: s_or_b32 [[OR:s[0-9]+]], [[VAL]], 4
define amdgpu_kernel void @widen_i16_constant32_load(i16 addrspace(6)* %arg) {
  %load = load i16, i16 addrspace(6)* %arg, align 4
  %add = add i16 %load, 999
  %or = or i16 %add, 4
  store i16 %or, i16 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}widen_i16_global_invariant_load:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_addk_i32 [[VAL]], 0x3e7
; GCN: s_or_b32 [[OR:s[0-9]+]], [[VAL]], 1
define amdgpu_kernel void @widen_i16_global_invariant_load(i16 addrspace(1)* %arg) {
  %load = load i16, i16 addrspace(1)* %arg, align 4, !invariant.load !0
  %add = add i16 %load, 999
  %or = or i16 %add, 1
  store i16 %or, i16 addrspace(1)* null
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

!0 = !{}
