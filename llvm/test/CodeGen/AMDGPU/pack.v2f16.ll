; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx901 -mattr=-flat-for-global,-fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s


; GCN-LABEL: {{^}}s_pack_v2f16:
; GFX9: s_load_dword [[VAL0:s[0-9]+]]
; GFX9: s_load_dword [[VAL1:s[0-9]+]]
; GFX9: s_pack_ll_b32_b16 [[PACKED:s[0-9]+]], [[VAL0]], [[VAL1]]
; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @s_pack_v2f16(i32 addrspace(2)* %in0, i32 addrspace(2)* %in1) #0 {
  %val0 = load volatile i32, i32 addrspace(2)* %in0
  %val1 = load volatile i32, i32 addrspace(2)* %in1
  %lo.i = trunc i32 %val0 to i16
  %hi.i = trunc i32 %val1 to i16
  %lo = bitcast i16 %lo.i to half
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}s_pack_v2f16_imm_lo:
; GFX9: s_load_dword [[VAL1:s[0-9]+]]
; GFX9: s_pack_ll_b32_b16 [[PACKED:s[0-9]+]], 0x1234, [[VAL1]]
; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @s_pack_v2f16_imm_lo(i32 addrspace(2)* %in1) #0 {
  %val1 = load i32, i32 addrspace(2)* %in1
  %hi.i = trunc i32 %val1 to i16
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half 0xH1234, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}s_pack_v2f16_imm_hi:
; GFX9: s_load_dword [[VAL0:s[0-9]+]]
; GFX9: s_pack_ll_b32_b16 [[PACKED:s[0-9]+]], [[VAL0]], 0x1234
; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @s_pack_v2f16_imm_hi(i32 addrspace(2)* %in0) #0 {
  %val0 = load i32, i32 addrspace(2)* %in0
  %lo.i = trunc i32 %val0 to i16
  %lo = bitcast i16 %lo.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half 0xH1234, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16:
; GFX9: flat_load_dword [[VAL0:v[0-9]+]]
; GFX9: flat_load_dword [[VAL1:v[0-9]+]]

; GFX9: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VAL0]]
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[ELT0]]
; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16(i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %lo.i = trunc i32 %val0 to i16
  %hi.i = trunc i32 %val1 to i16
  %lo = bitcast i16 %lo.i to half
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16_user:
; GFX9: flat_load_dword [[VAL0:v[0-9]+]]
; GFX9: flat_load_dword [[VAL1:v[0-9]+]]

; GFX9: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VAL0]]
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[ELT0]]

; GFX9: v_add_i32_e32 v{{[0-9]+}}, vcc, 9, [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16_user(i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %lo.i = trunc i32 %val0 to i16
  %hi.i = trunc i32 %val1 to i16
  %lo = bitcast i16 %lo.i to half
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  %foo = add i32 %vec.i32, 9
  store volatile i32 %foo, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16_imm_lo:
; GFX9-DAG: flat_load_dword [[VAL1:v[0-9]+]]

; GFX9-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x1234{{$}}
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[K]]
; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16_imm_lo(i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %hi.i = trunc i32 %val1 to i16
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half 0xH1234, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16_inline_imm_lo:
; GFX9-DAG: flat_load_dword [[VAL1:v[0-9]+]]

; GFX9-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x4400{{$}}
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[K]]

; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16_inline_imm_lo(i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %hi.i = trunc i32 %val1 to i16
  %hi = bitcast i16 %hi.i to half
  %vec.0 = insertelement <2 x half> undef, half 4.0, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half %hi, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16_imm_hi:
; GFX9-DAG: flat_load_dword [[VAL0:v[0-9]+]]

; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x1234
; GFX9: v_and_b32_e32 [[MASKED:v[0-9]+]], 0xffff, [[VAL0]]
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[K]], 16, [[MASKED]]

; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16_imm_hi(i32 addrspace(1)* %in0) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %lo.i = trunc i32 %val0 to i16
  %lo = bitcast i16 %lo.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half 0xH1234, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16_inline_f16imm_hi:
; GFX9-DAG: flat_load_dword [[VAL:v[0-9]+]]

; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x3c00
; GFX9: v_and_b32_e32 [[MASKED:v[0-9]+]], 0xffff, [[VAL]]
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[K]], 16, [[MASKED]]

; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16_inline_f16imm_hi(i32 addrspace(1)* %in0) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %lo.i = trunc i32 %val0 to i16
  %lo = bitcast i16 %lo.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half 1.0, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2f16_inline_imm_hi:
; GFX9: flat_load_dword [[VAL:v[0-9]+]]

; GFX9: v_and_b32_e32 [[MASKED:v[0-9]+]], 0xffff, [[VAL]]
; GFX9: v_lshl_or_b32 [[PACKED:v[0-9]+]], 64, 16, [[MASKED]]

; GFX9: ; use [[PACKED]]
define amdgpu_kernel void @v_pack_v2f16_inline_imm_hi(i32 addrspace(1)* %in0) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %lo.i = trunc i32 %val0 to i16
  %lo = bitcast i16 %lo.i to half
  %vec.0 = insertelement <2 x half> undef, half %lo, i32 0
  %vec.1 = insertelement <2 x half> %vec.0, half 0xH0040, i32 1
  %vec.i32 = bitcast <2 x half> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
