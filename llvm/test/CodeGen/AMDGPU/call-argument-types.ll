; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,MESA %s
; RUN: llc -march=amdgcn -mcpu=hawaii -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,MESA %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,VI,MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,HSA %s

declare void @external_void_func_i1(i1) #0
declare void @external_void_func_i1_signext(i1 signext) #0
declare void @external_void_func_i1_zeroext(i1 zeroext) #0

declare void @external_void_func_i8(i8) #0
declare void @external_void_func_i8_signext(i8 signext) #0
declare void @external_void_func_i8_zeroext(i8 zeroext) #0

declare void @external_void_func_i16(i16) #0
declare void @external_void_func_i16_signext(i16 signext) #0
declare void @external_void_func_i16_zeroext(i16 zeroext) #0

declare void @external_void_func_i32(i32) #0
declare void @external_void_func_i64(i64) #0

declare void @external_void_func_f16(half) #0
declare void @external_void_func_f32(float) #0
declare void @external_void_func_f64(double) #0

declare void @external_void_func_v2i16(<2 x i16>) #0
declare void @external_void_func_v2f16(<2 x half>) #0

declare void @external_void_func_v2i32(<2 x i32>) #0
declare void @external_void_func_v3i32(<3 x i32>) #0
declare void @external_void_func_v4i32(<4 x i32>) #0
declare void @external_void_func_v8i32(<8 x i32>) #0
declare void @external_void_func_v16i32(<16 x i32>) #0
declare void @external_void_func_v32i32(<32 x i32>) #0
declare void @external_void_func_v32i32_i32(<32 x i32>, i32) #0

; return value and argument
declare i32 @external_i32_func_i32(i32) #0

; Structs
declare void @external_void_func_struct_i8_i32({ i8, i32 }) #0
declare void @external_void_func_byval_struct_i8_i32({ i8, i32 }* byval) #0
declare void @external_void_func_sret_struct_i8_i32_byval_struct_i8_i32({ i8, i32 }* sret, { i8, i32 }* byval) #0

declare void @external_void_func_v16i8(<16 x i8>) #0


; FIXME: Should be passing -1
; GCN-LABEL: {{^}}test_call_external_void_func_i1_imm:
; MESA: s_mov_b32 s36, SCRATCH_RSRC_DWORD

; MESA-DAG: s_mov_b64 s[0:1], s[36:37]

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i1@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i1@rel32@hi+4
; GCN-DAG: v_mov_b32_e32 v0, 1{{$}}
; MESA-DAG: s_mov_b64 s[2:3], s[38:39]

; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i1_imm() #0 {
  call void @external_void_func_i1(i1 true)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i1_signext:
; MESA: s_mov_b32 s33, s3{{$}}
; HSA: s_mov_b32 s33, s9{{$}}

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i1_signext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i1_signext@rel32@hi+4
; GCN-NEXT: buffer_load_ubyte [[VAR:v[0-9]+]]
; HSA-NEXT: s_mov_b32 s4, s33
; HSA-NEXT: s_mov_b32 s32, s33

; MESA-DAG: s_mov_b32 s4, s33{{$}}
; MESA-DAG: s_mov_b32 s32, s33{{$}}

; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: v_bfe_i32 v0, v0, 0, 1
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i1_signext(i32) #0 {
  %var = load volatile i1, i1 addrspace(1)* undef
  call void @external_void_func_i1_signext(i1 %var)
  ret void
}

; FIXME: load should be scheduled before getpc
; GCN-LABEL: {{^}}test_call_external_void_func_i1_zeroext:
; MESA: s_mov_b32 s33, s3{{$}}

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i1_zeroext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i1_zeroext@rel32@hi+4
; GCN-NEXT: buffer_load_ubyte v0

; GCN-DAG: s_mov_b32 s4, s33{{$}}
; GCN-DAG: s_mov_b32 s32, s33{{$}}

; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: v_and_b32_e32 v0, 1, v0
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i1_zeroext(i32) #0 {
  %var = load volatile i1, i1 addrspace(1)* undef
  call void @external_void_func_i1_zeroext(i1 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i8_imm:
; MESA-DAG: s_mov_b32 s33, s3{{$}}

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i8@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i8@rel32@hi+4
; GCN-NEXT: v_mov_b32_e32 v0, 0x7b

; HSA-DAG: s_mov_b32 s4, s33{{$}}
; GCN-DAG: s_mov_b32 s32, s33{{$}}

; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i8_imm(i32) #0 {
  call void @external_void_func_i8(i8 123)
  ret void
}

; FIXME: don't wait before call
; GCN-LABEL: {{^}}test_call_external_void_func_i8_signext:
; HSA-DAG: s_mov_b32 s33, s9{{$}}
; MESA-DAG: s_mov_b32 s33, s3{{$}}

; GCN-DAG: buffer_load_sbyte v0
; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i8_signext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i8_signext@rel32@hi+4

; GCN-DAG: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s3

; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i8_signext(i32) #0 {
  %var = load volatile i8, i8 addrspace(1)* undef
  call void @external_void_func_i8_signext(i8 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i8_zeroext:
; MESA-DAG: s_mov_b32 s33, s3{{$}}
; HSA-DAG: s_mov_b32 s33, s9{{$}}

; GCN-DAG: buffer_load_ubyte v0
; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i8_zeroext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i8_zeroext@rel32@hi+4

; GCN-DAG: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s33

; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i8_zeroext(i32) #0 {
  %var = load volatile i8, i8 addrspace(1)* undef
  call void @external_void_func_i8_zeroext(i8 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i16_imm:
; GCN-DAG: v_mov_b32_e32 v0, 0x7b{{$}}

; GCN-DAG: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s33

; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_i16_imm() #0 {
  call void @external_void_func_i16(i16 123)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i16_signext:
; MESA-DAG: s_mov_b32 s33, s3{{$}}

; GCN-DAG: buffer_load_sshort v0
; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i16_signext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i16_signext@rel32@hi+4

; GCN-DAG: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s33

; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i16_signext(i32) #0 {
  %var = load volatile i16, i16 addrspace(1)* undef
  call void @external_void_func_i16_signext(i16 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i16_zeroext:
; MESA-DAG: s_mov_b32 s33, s3{{$}}


; GCN-DAG: buffer_load_ushort v0
; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i16_zeroext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i16_zeroext@rel32@hi+4

; GCN-DAG: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s33

; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i16_zeroext(i32) #0 {
  %var = load volatile i16, i16 addrspace(1)* undef
  call void @external_void_func_i16_zeroext(i16 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i32_imm:
; MESA-DAG: s_mov_b32 s33, s3{{$}}

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i32@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i32@rel32@hi+4
; GCN: v_mov_b32_e32 v0, 42
; GCN-DAG: s_mov_b32 s4, s33
; GCN-DAG: s_mov_b32 s32, s33

; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i32_imm(i32) #0 {
  call void @external_void_func_i32(i32 42)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i64_imm:
; GCN-DAG: s_movk_i32 [[K0:s[0-9]+]], 0x7b{{$}}
; GCN-DAG: s_mov_b32 [[K1:s[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v0, [[K0]]
; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i64@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i64@rel32@hi+4
; GCN-DAG: v_mov_b32_e32 v1, [[K1]]
; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i64_imm() #0 {
  call void @external_void_func_i64(i64 123)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_f16_imm:
; VI: v_mov_b32_e32 v0, 0x4400
; CI: v_mov_b32_e32 v0, 4.0
; GCN-NOT: v0
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_f16_imm() #0 {
  call void @external_void_func_f16(half 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_f32_imm:
; GCN: v_mov_b32_e32 v0, 4.0
; GCN-NOT: v0
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_f32_imm() #0 {
  call void @external_void_func_f32(float 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_f64_imm:
; GCN: v_mov_b32_e32 v0, 0{{$}}
; GCN: v_mov_b32_e32 v1, 0x40100000
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_f64_imm() #0 {
  call void @external_void_func_f64(double 4.0)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v2i16:
; GFX9: buffer_load_dword v0
; GFX9-NOT: v0
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2i16() #0 {
  %val = load <2 x i16>, <2 x i16> addrspace(1)* undef
  call void @external_void_func_v2i16(<2 x i16> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v2f16:
; GFX9: buffer_load_dword v0
; GFX9-NOT: v0
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2f16() #0 {
  %val = load <2 x half>, <2 x half> addrspace(1)* undef
  call void @external_void_func_v2f16(<2 x half> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v2i32:
; GCN: buffer_load_dwordx2 v[0:1]
; GCN: s_waitcnt
; GCN-NEXT: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2i32() #0 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* undef
  call void @external_void_func_v2i32(<2 x i32> %val)
  ret void
}

; FIXME: Passing 4th
; GCN-LABEL: {{^}}test_call_external_void_func_v3i32_imm:
; HSA-DAG: s_mov_b32 s33, s9
; MESA-DAG: s_mov_b32 s33, s3{{$}}

; GCN-DAG: v_mov_b32_e32 v0
; GCN-DAG: v_mov_b32_e32 v1
; GCN-DAG: v_mov_b32_e32 v2
; GCN-DAG: v_mov_b32_e32 v3

; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3i32_imm(i32) #0 {
  call void @external_void_func_v3i32(<3 x i32> <i32 3, i32 4, i32 5>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v4i32:
; GCN: buffer_load_dwordx4 v[0:3]
; GCN: s_waitcnt
; GCN-NEXT: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v4i32() #0 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* undef
  call void @external_void_func_v4i32(<4 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v8i32:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN: s_waitcnt
; GCN-NEXT: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v8i32() #0 {
  %ptr = load <8 x i32> addrspace(1)*, <8 x i32> addrspace(1)* addrspace(2)* undef
  %val = load <8 x i32>, <8 x i32> addrspace(1)* %ptr
  call void @external_void_func_v8i32(<8 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v16i32:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN: s_waitcnt
; GCN-NEXT: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v16i32() #0 {
  %ptr = load <16 x i32> addrspace(1)*, <16 x i32> addrspace(1)* addrspace(2)* undef
  %val = load <16 x i32>, <16 x i32> addrspace(1)* %ptr
  call void @external_void_func_v16i32(<16 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v32i32:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN-DAG: buffer_load_dwordx4 v[16:19], off
; GCN-DAG: buffer_load_dwordx4 v[20:23], off
; GCN-DAG: buffer_load_dwordx4 v[24:27], off
; GCN-DAG: buffer_load_dwordx4 v[28:31], off
; GCN: s_waitcnt
; GCN-NEXT: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v32i32() #0 {
  %ptr = load <32 x i32> addrspace(1)*, <32 x i32> addrspace(1)* addrspace(2)* undef
  %val = load <32 x i32>, <32 x i32> addrspace(1)* %ptr
  call void @external_void_func_v32i32(<32 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v32i32_i32:
; HSA-DAG: s_mov_b32 s33, s9
; HSA-DAG: s_add_u32 [[SP_REG:s[0-9]+]], s33, 0x100{{$}}

; MESA-DAG: s_mov_b32 s33, s3{{$}}
; MESA-DAG: s_add_u32 [[SP_REG:s[0-9]+]], s33, 0x100{{$}}

; GCN-DAG: buffer_load_dword [[VAL1:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN-DAG: buffer_load_dwordx4 v[16:19], off
; GCN-DAG: buffer_load_dwordx4 v[20:23], off
; GCN-DAG: buffer_load_dwordx4 v[24:27], off
; GCN-DAG: buffer_load_dwordx4 v[28:31], off

; GCN: buffer_store_dword [[VAL1]], off, s[{{[0-9]+}}:{{[0-9]+}}], [[SP_REG]] offset:4{{$}}
; GCN: s_waitcnt
; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_v32i32_i32(i32) #0 {
  %ptr0 = load <32 x i32> addrspace(1)*, <32 x i32> addrspace(1)* addrspace(2)* undef
  %val0 = load <32 x i32>, <32 x i32> addrspace(1)* %ptr0
  %val1 = load i32, i32 addrspace(1)* undef
  call void @external_void_func_v32i32_i32(<32 x i32> %val0, i32 %val1)
  ret void
}

; FIXME: No wait after call
; GCN-LABEL: {{^}}test_call_external_i32_func_i32_imm:
; GCN: v_mov_b32_e32 v0, 42
; GCN: s_swappc_b64 s[30:31],
; GCN-NEXT: s_waitcnt lgkmcnt(0)
; GCN-NEXT: buffer_store_dword v0, off, s[36:39], 0
define amdgpu_kernel void @test_call_external_i32_func_i32_imm(i32 addrspace(1)* %out) #0 {
  %val = call i32 @external_i32_func_i32(i32 42)
  store volatile i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_struct_i8_i32:
; GCN: buffer_load_ubyte v0, off
; GCN: buffer_load_dword v1, off
; GCN: s_waitcnt vmcnt(0)
; GCN-NEXT: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_struct_i8_i32() #0 {
  %ptr0 = load { i8, i32 } addrspace(1)*, { i8, i32 } addrspace(1)* addrspace(2)* undef
  %val = load { i8, i32 }, { i8, i32 } addrspace(1)* %ptr0
  call void @external_void_func_struct_i8_i32({ i8, i32 } %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_byval_struct_i8_i32:
; GCN-DAG: s_add_u32 [[SP:s[0-9]+]], s33, 0x400{{$}}

; GCN-DAG: v_mov_b32_e32 [[VAL0:v[0-9]+]], 3
; GCN-DAG: v_mov_b32_e32 [[VAL1:v[0-9]+]], 8
; MESA-DAG: buffer_store_byte [[VAL0]], off, s[36:39], s33 offset:8
; MESA-DAG: buffer_store_dword [[VAL1]], off, s[36:39], s33 offset:12

; HSA-DAG: buffer_store_byte [[VAL0]], off, s[0:3], s33 offset:8
; HSA-DAG: buffer_store_dword [[VAL1]], off, s[0:3], s33 offset:12

; GCN: s_add_u32 [[SP]], [[SP]], 0x200

; HSA: buffer_load_dword [[RELOAD_VAL0:v[0-9]+]], off, s[0:3], s33 offset:8
; HSA: buffer_load_dword [[RELOAD_VAL1:v[0-9]+]], off, s[0:3], s33 offset:12

; HSA: buffer_store_dword [[RELOAD_VAL1]], off, s[0:3], [[SP]] offset:8
; HSA: buffer_store_dword [[RELOAD_VAL0]], off, s[0:3], [[SP]] offset:4


; MESA: buffer_load_dword [[RELOAD_VAL0:v[0-9]+]], off, s[36:39], s33 offset:8
; MESA: buffer_load_dword [[RELOAD_VAL1:v[0-9]+]], off, s[36:39], s33 offset:12

; MESA: buffer_store_dword [[RELOAD_VAL1]], off, s[36:39], [[SP]] offset:8
; MESA: buffer_store_dword [[RELOAD_VAL0]], off, s[36:39], [[SP]] offset:4

; GCN-NEXT: s_swappc_b64
; GCN-NEXT: s_sub_u32 [[SP]], [[SP]], 0x200
define amdgpu_kernel void @test_call_external_void_func_byval_struct_i8_i32() #0 {
  %val = alloca { i8, i32 }, align 4
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %val, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %val, i32 0, i32 1
  store i8 3, i8* %gep0
  store i32 8, i32* %gep1
  call void @external_void_func_byval_struct_i8_i32({ i8, i32 }* %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_sret_struct_i8_i32_byval_struct_i8_i32:
; MESA-DAG: s_add_u32 [[SP:s[0-9]+]], [[FP_REG:s[0-9]+]], 0x600{{$}}
; HSA-DAG: s_add_u32 [[SP:s[0-9]+]], [[FP_REG:s[0-9]+]], 0x600{{$}}

; GCN-DAG: v_mov_b32_e32 [[VAL0:v[0-9]+]], 3
; GCN-DAG: v_mov_b32_e32 [[VAL1:v[0-9]+]], 8
; GCN-DAG: buffer_store_byte [[VAL0]], off, s{{\[[0-9]+:[0-9]+\]}}, [[FP_REG]] offset:8
; GCN-DAG: buffer_store_dword [[VAL1]], off, s{{\[[0-9]+:[0-9]+\]}}, [[FP_REG]] offset:12

; GCN-DAG: buffer_load_dword [[RELOAD_VAL0:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, [[FP_REG]] offset:8
; GCN-DAG: buffer_load_dword [[RELOAD_VAL1:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, [[FP_REG]] offset:12

; GCN-DAG: s_add_u32 [[SP]], [[SP]], 0x200
; GCN: buffer_store_dword [[RELOAD_VAL1]], off, s{{\[[0-9]+:[0-9]+\]}}, [[SP]] offset:8
; GCN: buffer_store_dword [[RELOAD_VAL0]], off, s{{\[[0-9]+:[0-9]+\]}}, [[SP]] offset:4
; GCN-NEXT: s_swappc_b64
; GCN-DAG: buffer_load_ubyte [[LOAD_OUT_VAL0:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, [[FP_REG]] offset:16
; GCN-DAG: buffer_load_dword [[LOAD_OUT_VAL1:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, [[FP_REG]] offset:20
; GCN: s_sub_u32 [[SP]], [[SP]], 0x200

; GCN: buffer_store_byte [[LOAD_OUT_VAL0]], off
; GCN: buffer_store_dword [[LOAD_OUT_VAL1]], off
define amdgpu_kernel void @test_call_external_void_func_sret_struct_i8_i32_byval_struct_i8_i32(i32) #0 {
  %in.val = alloca { i8, i32 }, align 4
  %out.val = alloca { i8, i32 }, align 4
  %in.gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %in.val, i32 0, i32 0
  %in.gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %in.val, i32 0, i32 1
  store i8 3, i8* %in.gep0
  store i32 8, i32* %in.gep1
  call void @external_void_func_sret_struct_i8_i32_byval_struct_i8_i32({ i8, i32 }* %out.val, { i8, i32 }* %in.val)
  %out.gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %out.val, i32 0, i32 0
  %out.gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 }* %out.val, i32 0, i32 1
  %out.val0 = load i8, i8* %out.gep0
  %out.val1 = load i32, i32* %out.gep1

  store volatile i8 %out.val0, i8 addrspace(1)* undef
  store volatile i32 %out.val1, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v16i8:
define amdgpu_kernel void @test_call_external_void_func_v16i8() #0 {
  %ptr = load <16 x i8> addrspace(1)*, <16 x i8> addrspace(1)* addrspace(2)* undef
  %val = load <16 x i8>, <16 x i8> addrspace(1)* %ptr
  call void @external_void_func_v16i8(<16 x i8> %val)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
