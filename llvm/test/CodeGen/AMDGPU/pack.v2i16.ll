; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx901 -mattr=-flat-for-global,+fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX9-DENORM %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx901 -mattr=-flat-for-global,-fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX9-FLUSH %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s


; GCN-LABEL: {{^}}s_pack_v2i16:
; GFX9: s_load_dword [[VAL0:s[0-9]+]]
; GFX9: s_load_dword [[VAL1:s[0-9]+]]
; GFX9: s_pack_ll_b32_b16 [[PACKED:s[0-9]+]], [[VAL0]], [[VAL1]]
; GFX9: ; use [[PACKED]]
define void @s_pack_v2i16(i32 addrspace(2)* %in0, i32 addrspace(2)* %in1) #0 {
  %val0 = load volatile i32, i32 addrspace(2)* %in0
  %val1 = load volatile i32, i32 addrspace(2)* %in1
  %lo = trunc i32 %val0 to i16
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}s_pack_v2i16_imm_lo:
; GFX9: s_load_dword [[VAL1:s[0-9]+]]
; GFX9: s_pack_ll_b32_b16 [[PACKED:s[0-9]+]], 0x1c8, [[VAL1]]
; GFX9: ; use [[PACKED]]
define void @s_pack_v2i16_imm_lo(i32 addrspace(2)* %in1) #0 {
  %val1 = load i32, i32 addrspace(2)* %in1
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 456, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}s_pack_v2i16_imm_hi:
; GFX9: s_load_dword [[VAL0:s[0-9]+]]
; GFX9: s_pack_ll_b32_b16 [[PACKED:s[0-9]+]], [[VAL0]], 0x1c8
; GFX9: ; use [[PACKED]]
define void @s_pack_v2i16_imm_hi(i32 addrspace(2)* %in0) #0 {
  %val0 = load i32, i32 addrspace(2)* %in0
  %lo = trunc i32 %val0 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 456, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32

  call void asm sideeffect "; use $0", "s"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2i16:
; GFX9: flat_load_dword [[VAL0:v[0-9]+]]
; GFX9: flat_load_dword [[VAL1:v[0-9]+]]
; GFX9-DENORM: v_pack_b32_f16 [[PACKED:v[0-9]+]], [[VAL0]], [[VAL1]]

; GFX9-FLUSH: v_and_b32_e32 [[MASKED:v[0-9]+]], 0xffff, [[VAL0]]
; GFX9-FLUSH: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[MASKED]]
; GFX9: ; use [[PACKED]]
define void @v_pack_v2i16(i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %lo = trunc i32 %val0 to i16
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2i16_user:
; GFX9: flat_load_dword [[VAL0:v[0-9]+]]
; GFX9: flat_load_dword [[VAL1:v[0-9]+]]
; GFX9-DENORM: v_pack_b32_f16 [[PACKED:v[0-9]+]], [[VAL0]], [[VAL1]]

; GFX9-FLUSH: v_and_b32_e32 [[MASKED:v[0-9]+]], 0xffff, [[VAL0]]
; GFX9-FLUSH: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[MASKED]]

; GFX9: v_add_i32_e32 v{{[0-9]+}}, vcc, 9, [[PACKED]]
define void @v_pack_v2i16_user(i32 addrspace(1)* %in0, i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %lo = trunc i32 %val0 to i16
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  %foo = add i32 %vec.i32, 9
  store volatile i32 %foo, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2i16_imm_lo:
; GFX9-DAG: flat_load_dword [[VAL1:v[0-9]+]]
; GFX9-DENORM-DAG: s_movk_i32 [[K:s[0-9]+]], 0x7b{{$}}
; GFX9-DENORM: v_pack_b32_f16 [[PACKED:v[0-9]+]], [[K]], [[VAL1]]

; GFX9-FLUSH-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x7b{{$}}
; GFX9-FLUSH: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, [[K]]

; GFX9: ; use [[PACKED]]
define void @v_pack_v2i16_imm_lo(i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 123, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2i16_inline_imm_lo:
; GFX9: flat_load_dword [[VAL1:v[0-9]+]]
; GFX9-DENORM: v_pack_b32_f16 [[PACKED:v[0-9]+]], 64, [[VAL1]]

; GFX9-FLUSH: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[VAL1]], 16, 64
; GFX9: ; use [[PACKED]]
define void @v_pack_v2i16_inline_imm_lo(i32 addrspace(1)* %in1) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in1.gep = getelementptr inbounds i32, i32 addrspace(1)* %in1, i64 %tid.ext
  %val1 = load volatile i32, i32 addrspace(1)* %in1.gep
  %hi = trunc i32 %val1 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 64, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 %hi, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2i16_imm_hi:
; GFX9-DAG: flat_load_dword [[VAL0:v[0-9]+]]
; GFX9-DENORM: v_pack_b32_f16 [[PACKED:v[0-9]+]], [[VAL0]], [[K]]

; GFX9-FLUSH-DAG: s_movk_i32 [[K:s[0-9]+]], 0x7b{{$}}
; GFX9-FLUSH: v_lshl_or_b32 [[PACKED:v[0-9]+]], [[K]], 16, [[VAL0]]

; GFX9: ; use [[PACKED]]
define void @v_pack_v2i16_imm_hi(i32 addrspace(1)* %in0) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %lo = trunc i32 %val0 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 123, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

; GCN-LABEL: {{^}}v_pack_v2i16_inline_imm_hi:
; GFX9: flat_load_dword [[VAL:v[0-9]+]]
; GFX9-DENORM: v_pack_b32_f16 [[PACKED:v[0-9]+]], [[VAL]], 7
; GFX9-FLUSH: v_lshl_or_b32 [[PACKED:v[0-9]+]], 7, 16, [[VAL0]]
; GFX9: ; use [[PACKED]]
define void @v_pack_v2i16_inline_imm_hi(i32 addrspace(1)* %in0) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in0.gep = getelementptr inbounds i32, i32 addrspace(1)* %in0, i64 %tid.ext
  %val0 = load volatile i32, i32 addrspace(1)* %in0.gep
  %lo = trunc i32 %val0 to i16
  %vec.0 = insertelement <2 x i16> undef, i16 %lo, i32 0
  %vec.1 = insertelement <2 x i16> %vec.0, i16 7, i32 1
  %vec.i32 = bitcast <2 x i16> %vec.1 to i32
  call void asm sideeffect "; use $0", "v"(i32 %vec.i32) #0
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
