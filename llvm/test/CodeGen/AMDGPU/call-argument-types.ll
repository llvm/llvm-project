; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,MESA %s
; RUN: llc -march=amdgcn -mcpu=hawaii -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,MESA %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,VI,MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global -amdgpu-scalarize-global-loads=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,HSA %s

declare hidden void @external_void_func_i1(i1) #0
declare hidden void @external_void_func_i1_signext(i1 signext) #0
declare hidden void @external_void_func_i1_zeroext(i1 zeroext) #0

declare hidden void @external_void_func_i8(i8) #0
declare hidden void @external_void_func_i8_signext(i8 signext) #0
declare hidden void @external_void_func_i8_zeroext(i8 zeroext) #0

declare hidden void @external_void_func_i16(i16) #0
declare hidden void @external_void_func_i16_signext(i16 signext) #0
declare hidden void @external_void_func_i16_zeroext(i16 zeroext) #0

declare hidden void @external_void_func_i32(i32) #0
declare hidden void @external_void_func_i64(i64) #0
declare hidden void @external_void_func_v2i64(<2 x i64>) #0
declare hidden void @external_void_func_v3i64(<3 x i64>) #0
declare hidden void @external_void_func_v4i64(<4 x i64>) #0

declare hidden void @external_void_func_f16(half) #0
declare hidden void @external_void_func_f32(float) #0
declare hidden void @external_void_func_f64(double) #0
declare hidden void @external_void_func_v2f32(<2 x float>) #0
declare hidden void @external_void_func_v2f64(<2 x double>) #0
declare hidden void @external_void_func_v3f32(<3 x float>) #0
declare hidden void @external_void_func_v3f64(<3 x double>) #0
declare hidden void @external_void_func_v5f32(<5 x float>) #0

declare hidden void @external_void_func_v2i16(<2 x i16>) #0
declare hidden void @external_void_func_v2f16(<2 x half>) #0
declare hidden void @external_void_func_v3i16(<3 x i16>) #0
declare hidden void @external_void_func_v3f16(<3 x half>) #0
declare hidden void @external_void_func_v4i16(<4 x i16>) #0
declare hidden void @external_void_func_v4f16(<4 x half>) #0

declare hidden void @external_void_func_v2i32(<2 x i32>) #0
declare hidden void @external_void_func_v3i32(<3 x i32>) #0
declare hidden void @external_void_func_v3i32_i32(<3 x i32>, i32) #0
declare hidden void @external_void_func_v4i32(<4 x i32>) #0
declare hidden void @external_void_func_v5i32(<5 x i32>) #0
declare hidden void @external_void_func_v8i32(<8 x i32>) #0
declare hidden void @external_void_func_v16i32(<16 x i32>) #0
declare hidden void @external_void_func_v32i32(<32 x i32>) #0
declare hidden void @external_void_func_v32i32_i32(<32 x i32>, i32) #0

; return value and argument
declare hidden i32 @external_i32_func_i32(i32) #0

; Structs
declare hidden void @external_void_func_struct_i8_i32({ i8, i32 }) #0
declare hidden void @external_void_func_byval_struct_i8_i32({ i8, i32 } addrspace(5)* byval) #0
declare hidden void @external_void_func_sret_struct_i8_i32_byval_struct_i8_i32({ i8, i32 } addrspace(5)* sret, { i8, i32 } addrspace(5)* byval) #0

declare hidden void @external_void_func_v16i8(<16 x i8>) #0


; FIXME: Should be passing -1
; GCN-LABEL: {{^}}test_call_external_void_func_i1_imm:
; MESA: s_mov_b32 s36, SCRATCH_RSRC_DWORD

; MESA-DAG: s_mov_b64 s[0:1], s[36:37]

; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i1@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i1@rel32@hi+4
; GCN-DAG: v_mov_b32_e32 v0, 1{{$}}
; MESA-DAG: s_mov_b64 s[2:3], s[38:39]

; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i1_imm() #0 {
  call void @external_void_func_i1(i1 true)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i1_signext:

; HSA: buffer_load_ubyte [[VAR:v[0-9]+]]
; HSA: s_mov_b32 s32, 0
; MESA-DAG: buffer_load_ubyte [[VAR:v[0-9]+]]
; MESA-DAG: s_mov_b32 s32, 0{{$}}

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i1_signext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i1_signext@rel32@hi+4

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

; HSA: buffer_load_ubyte v0
; HSA-DAG: s_mov_b32 s32, 0{{$}}

; MESA: buffer_load_ubyte v0
; MESA-DAG: s_mov_b32 s32, 0{{$}}

; GCN: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i1_zeroext@rel32@lo+4
; GCN-NEXT: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i1_zeroext@rel32@hi+4


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

; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i8@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i8@rel32@hi+4
; GCN-DAG: v_mov_b32_e32 v0, 0x7b

; GCN-DAG: s_mov_b32 s32, 0{{$}}

; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i8_imm(i32) #0 {
  call void @external_void_func_i8(i8 123)
  ret void
}

; FIXME: don't wait before call
; GCN-LABEL: {{^}}test_call_external_void_func_i8_signext:

; GCN-DAG: buffer_load_sbyte v0
; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i8_signext@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i8_signext@rel32@hi+4

; GCN-DAG: s_mov_b32 s32, 0

; GCN-NOT: s_waitcnt
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i8_signext(i32) #0 {
  %var = load volatile i8, i8 addrspace(1)* undef
  call void @external_void_func_i8_signext(i8 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i8_zeroext:

; GCN-DAG: buffer_load_ubyte v0
; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i8_zeroext@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i8_zeroext@rel32@hi+4

; GCN-DAG: s_mov_b32 s32, 0

; GCN-NOT: s_waitcnt
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i8_zeroext(i32) #0 {
  %var = load volatile i8, i8 addrspace(1)* undef
  call void @external_void_func_i8_zeroext(i8 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i16_imm:
; GCN-DAG: v_mov_b32_e32 v0, 0x7b{{$}}

; GCN-DAG: s_mov_b32 s32, 0

; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_i16_imm() #0 {
  call void @external_void_func_i16(i16 123)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i16_signext:

; GCN-DAG: buffer_load_sshort v0
; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i16_signext@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i16_signext@rel32@hi+4

; GCN-DAG: s_mov_b32 s32, 0

; GCN-NOT: s_waitcnt
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i16_signext(i32) #0 {
  %var = load volatile i16, i16 addrspace(1)* undef
  call void @external_void_func_i16_signext(i16 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i16_zeroext:

; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i16_zeroext@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i16_zeroext@rel32@hi+4

; GCN-DAG: s_mov_b32 s32, 0

; GCN-NOT: s_waitcnt
; GCN-NEXT: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i16_zeroext(i32) #0 {
  %var = load volatile i16, i16 addrspace(1)* undef
  call void @external_void_func_i16_zeroext(i16 %var)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i32_imm:

; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i32@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i32@rel32@hi+4
; GCN-DAG: v_mov_b32_e32 v0, 42
; GCN-DAG: s_mov_b32 s32, 0

; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i32_imm(i32) #0 {
  call void @external_void_func_i32(i32 42)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_i64_imm:
; GCN-DAG: v_mov_b32_e32 v0, 0x7b{{$}}
; GCN-DAG: v_mov_b32_e32 v1, 0{{$}}
; GCN-DAG: s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-DAG: s_add_u32 s[[PC_LO]], s[[PC_LO]], external_void_func_i64@rel32@lo+4
; GCN-DAG: s_addc_u32 s[[PC_HI]], s[[PC_HI]], external_void_func_i64@rel32@hi+4
; GCN: s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_i64_imm() #0 {
  call void @external_void_func_i64(i64 123)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v2i64:
; GCN: buffer_load_dwordx4 v[0:3]
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2i64() #0 {
  %val = load <2 x i64>, <2 x i64> addrspace(1)* null
  call void @external_void_func_v2i64(<2 x i64> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v2i64_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1
; GCN-DAG: v_mov_b32_e32 v1, 2
; GCN-DAG: v_mov_b32_e32 v2, 3
; GCN-DAG: v_mov_b32_e32 v3, 4
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2i64_imm() #0 {
  call void @external_void_func_v2i64(<2 x i64> <i64 8589934593, i64 17179869187>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3i64:
; GCN: buffer_load_dwordx4 v[0:3]
; GCN: v_mov_b32_e32 v4, 1
; GCN: v_mov_b32_e32 v5, 2
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3i64() #0 {
  %load = load <2 x i64>, <2 x i64> addrspace(1)* null
  %val = shufflevector <2 x i64> %load, <2 x i64> <i64 8589934593, i64 undef>, <3 x i32> <i32 0, i32 1, i32 2>

  call void @external_void_func_v3i64(<3 x i64> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v4i64:
; GCN: buffer_load_dwordx4 v[0:3]
; GCN-DAG: v_mov_b32_e32 v4, 1
; GCN-DAG: v_mov_b32_e32 v5, 2
; GCN-DAG: v_mov_b32_e32 v6, 3
; GCN-DAG: v_mov_b32_e32 v7, 4

; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v4i64() #0 {
  %load = load <2 x i64>, <2 x i64> addrspace(1)* null
  %val = shufflevector <2 x i64> %load, <2 x i64> <i64 8589934593, i64 17179869187>, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  call void @external_void_func_v4i64(<4 x i64> %val)
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

; GCN-LABEL: {{^}}test_call_external_void_func_v2f32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2f32_imm() #0 {
  call void @external_void_func_v2f32(<2 x float> <float 1.0, float 2.0>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3f32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 4.0
; GCN-NOT: v3,
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3f32_imm() #0 {
  call void @external_void_func_v3f32(<3 x float> <float 1.0, float 2.0, float 4.0>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v5f32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 4.0
; GCN-DAG: v_mov_b32_e32 v3, -1.0
; GCN-DAG: v_mov_b32_e32 v4, 0.5
; GCN-NOT: v5,
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v5f32_imm() #0 {
  call void @external_void_func_v5f32(<5 x float> <float 1.0, float 2.0, float 4.0, float -1.0, float 0.5>)
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

; GCN-LABEL: {{^}}test_call_external_void_func_v2f64_imm:
; GCN: v_mov_b32_e32 v0, 0{{$}}
; GCN: v_mov_b32_e32 v1, 2.0
; GCN: v_mov_b32_e32 v2, 0{{$}}
; GCN: v_mov_b32_e32 v3, 0x40100000
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2f64_imm() #0 {
  call void @external_void_func_v2f64(<2 x double> <double 2.0, double 4.0>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3f64_imm:
; GCN-DAG: v_mov_b32_e32 v0, 0{{$}}
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 0{{$}}
; GCN-DAG: v_mov_b32_e32 v3, 0x40100000
; GCN-DAG: v_mov_b32_e32 v4, 0{{$}}
; GCN-DAG: v_mov_b32_e32 v5, 0x40200000
; GCN-DAG: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3f64_imm() #0 {
  call void @external_void_func_v3f64(<3 x double> <double 2.0, double 4.0, double 8.0>)
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

; GCN-LABEL: {{^}}test_call_external_void_func_v3i16:
; GFX9: buffer_load_dwordx2 v[0:1]
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3i16() #0 {
  %val = load <3 x i16>, <3 x i16> addrspace(1)* undef
  call void @external_void_func_v3i16(<3 x i16> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3f16:
; GFX9: buffer_load_dwordx2 v[0:1]
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3f16() #0 {
  %val = load <3 x half>, <3 x half> addrspace(1)* undef
  call void @external_void_func_v3f16(<3 x half> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3i16_imm:
; GFX9: v_mov_b32_e32 v0, 0x20001
; GFX9: v_mov_b32_e32 v1, 3
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3i16_imm() #0 {
  call void @external_void_func_v3i16(<3 x i16> <i16 1, i16 2, i16 3>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3f16_imm:
; GFX9: v_mov_b32_e32 v0, 0x40003c00
; GFX9: v_mov_b32_e32 v1, 0x4400
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3f16_imm() #0 {
  call void @external_void_func_v3f16(<3 x half> <half 1.0, half 2.0, half 4.0>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v4i16:
; GFX9: buffer_load_dwordx2 v[0:1]
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v4i16() #0 {
  %val = load <4 x i16>, <4 x i16> addrspace(1)* undef
  call void @external_void_func_v4i16(<4 x i16> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v4i16_imm:
; GFX9-DAG: v_mov_b32_e32 v0, 0x20001
; GFX9-DAG: v_mov_b32_e32 v1, 0x40003
; GFX9: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v4i16_imm() #0 {
  call void @external_void_func_v4i16(<4 x i16> <i16 1, i16 2, i16 3, i16 4>)
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
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2i32() #0 {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* undef
  call void @external_void_func_v2i32(<2 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v2i32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1
; GCN-DAG: v_mov_b32_e32 v1, 2
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v2i32_imm() #0 {
  call void @external_void_func_v2i32(<2 x i32> <i32 1, i32 2>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3i32_imm: {{.*}}

; GCN-NOT: v3
; GCN-DAG: v_mov_b32_e32 v0, 3
; GCN-DAG: v_mov_b32_e32 v1, 4
; GCN-DAG: v_mov_b32_e32 v2, 5

; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v3i32_imm(i32) #0 {
  call void @external_void_func_v3i32(<3 x i32> <i32 3, i32 4, i32 5>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v3i32_i32:
; GCN-DAG: v_mov_b32_e32 v0, 3
; GCN-DAG: v_mov_b32_e32 v1, 4
; GCN-DAG: v_mov_b32_e32 v2, 5
; GCN-DAG: v_mov_b32_e32 v3, 6
define amdgpu_kernel void @test_call_external_void_func_v3i32_i32(i32) #0 {
  call void @external_void_func_v3i32_i32(<3 x i32> <i32 3, i32 4, i32 5>, i32 6)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v4i32:
; GCN: buffer_load_dwordx4 v[0:3]
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v4i32() #0 {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* undef
  call void @external_void_func_v4i32(<4 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v4i32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1
; GCN-DAG: v_mov_b32_e32 v1, 2
; GCN-DAG: v_mov_b32_e32 v2, 3
; GCN-DAG: v_mov_b32_e32 v3, 4
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v4i32_imm() #0 {
  call void @external_void_func_v4i32(<4 x i32> <i32 1, i32 2, i32 3, i32 4>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v5i32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1
; GCN-DAG: v_mov_b32_e32 v1, 2
; GCN-DAG: v_mov_b32_e32 v2, 3
; GCN-DAG: v_mov_b32_e32 v3, 4
; GCN-DAG: v_mov_b32_e32 v4, 5
; GCN-NOT v5,
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v5i32_imm() #0 {
  call void @external_void_func_v5i32(<5 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v8i32:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v8i32() #0 {
  %ptr = load <8 x i32> addrspace(1)*, <8 x i32> addrspace(1)* addrspace(4)* undef
  %val = load <8 x i32>, <8 x i32> addrspace(1)* %ptr
  call void @external_void_func_v8i32(<8 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v8i32_imm:
; GCN-DAG: v_mov_b32_e32 v0, 1
; GCN-DAG: v_mov_b32_e32 v1, 2
; GCN-DAG: v_mov_b32_e32 v2, 3
; GCN-DAG: v_mov_b32_e32 v3, 4
; GCN-DAG: v_mov_b32_e32 v4, 5
; GCN-DAG: v_mov_b32_e32 v5, 6
; GCN-DAG: v_mov_b32_e32 v6, 7
; GCN-DAG: v_mov_b32_e32 v7, 8
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v8i32_imm() #0 {
  call void @external_void_func_v8i32(<8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v16i32:
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v16i32() #0 {
  %ptr = load <16 x i32> addrspace(1)*, <16 x i32> addrspace(1)* addrspace(4)* undef
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
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_v32i32() #0 {
  %ptr = load <32 x i32> addrspace(1)*, <32 x i32> addrspace(1)* addrspace(4)* undef
  %val = load <32 x i32>, <32 x i32> addrspace(1)* %ptr
  call void @external_void_func_v32i32(<32 x i32> %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v32i32_i32:
; HSA-NOT: s_add_u32 s32

; MESA-NOT: s_add_u32 s32

; GCN-DAG: buffer_load_dword [[VAL1:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GCN-DAG: buffer_load_dwordx4 v[0:3], off
; GCN-DAG: buffer_load_dwordx4 v[4:7], off
; GCN-DAG: buffer_load_dwordx4 v[8:11], off
; GCN-DAG: buffer_load_dwordx4 v[12:15], off
; GCN-DAG: buffer_load_dwordx4 v[16:19], off
; GCN-DAG: buffer_load_dwordx4 v[20:23], off
; GCN-DAG: buffer_load_dwordx4 v[24:27], off
; GCN-DAG: buffer_load_dwordx4 v[28:31], off

; GCN: s_waitcnt
; GCN: buffer_store_dword [[VAL1]], off, s[{{[0-9]+}}:{{[0-9]+}}], s32{{$}}
; GCN: s_swappc_b64
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @test_call_external_void_func_v32i32_i32(i32) #0 {
  %ptr0 = load <32 x i32> addrspace(1)*, <32 x i32> addrspace(1)* addrspace(4)* undef
  %val0 = load <32 x i32>, <32 x i32> addrspace(1)* %ptr0
  %val1 = load i32, i32 addrspace(1)* undef
  call void @external_void_func_v32i32_i32(<32 x i32> %val0, i32 %val1)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i32_func_i32_imm:
; GCN: v_mov_b32_e32 v0, 42
; GCN: s_swappc_b64 s[30:31],
; GCN-NOT: s_waitcnt
; GCN: buffer_store_dword v0, off, s[36:39], 0
define amdgpu_kernel void @test_call_external_i32_func_i32_imm(i32 addrspace(1)* %out) #0 {
  %val = call i32 @external_i32_func_i32(i32 42)
  store volatile i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_struct_i8_i32:
; GCN: buffer_load_ubyte v0, off
; GCN: buffer_load_dword v1, off
; GCN-NOT: s_waitcnt
; GCN: s_swappc_b64
define amdgpu_kernel void @test_call_external_void_func_struct_i8_i32() #0 {
  %ptr0 = load { i8, i32 } addrspace(1)*, { i8, i32 } addrspace(1)* addrspace(4)* undef
  %val = load { i8, i32 }, { i8, i32 } addrspace(1)* %ptr0
  call void @external_void_func_struct_i8_i32({ i8, i32 } %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_byval_struct_i8_i32:
; GCN-DAG: v_mov_b32_e32 [[VAL0:v[0-9]+]], 3
; GCN-DAG: v_mov_b32_e32 [[VAL1:v[0-9]+]], 8
; MESA-DAG: buffer_store_byte [[VAL0]], off, s[36:39], 0 offset:8
; MESA-DAG: buffer_store_dword [[VAL1]], off, s[36:39], 0 offset:12

; HSA-DAG: buffer_store_byte [[VAL0]], off, s[0:3], 0 offset:8
; HSA-DAG: buffer_store_dword [[VAL1]], off, s[0:3], 0 offset:12

; HSA: buffer_load_dword [[RELOAD_VAL0:v[0-9]+]], off, s[0:3], 0 offset:8
; HSA: buffer_load_dword [[RELOAD_VAL1:v[0-9]+]], off, s[0:3], 0 offset:12

; MESA: buffer_load_dword [[RELOAD_VAL0:v[0-9]+]], off, s[36:39], 0 offset:8
; MESA: buffer_load_dword [[RELOAD_VAL1:v[0-9]+]], off, s[36:39], 0 offset:12

; GCN-DAG: s_movk_i32 [[SP:s[0-9]+]], 0x400{{$}}

; HSA-DAG: buffer_store_dword [[RELOAD_VAL0]], off, s[0:3], [[SP]]{{$}}
; HSA-DAG: buffer_store_dword [[RELOAD_VAL1]], off, s[0:3], [[SP]] offset:4

; MESA-DAG: buffer_store_dword [[RELOAD_VAL0]], off, s[36:39], [[SP]]{{$}}
; MESA-DAG: buffer_store_dword [[RELOAD_VAL1]], off, s[36:39], [[SP]] offset:4

; GCN-NEXT: s_swappc_b64
; GCN-NOT: [[SP]]
define amdgpu_kernel void @test_call_external_void_func_byval_struct_i8_i32() #0 {
  %val = alloca { i8, i32 }, align 4, addrspace(5)
  %gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %val, i32 0, i32 0
  %gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %val, i32 0, i32 1
  store i8 3, i8 addrspace(5)* %gep0
  store i32 8, i32 addrspace(5)* %gep1
  call void @external_void_func_byval_struct_i8_i32({ i8, i32 } addrspace(5)* %val)
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_sret_struct_i8_i32_byval_struct_i8_i32:
; GCN-DAG: s_movk_i32 [[SP:s[0-9]+]], 0x800{{$}}

; GCN-DAG: v_mov_b32_e32 [[VAL0:v[0-9]+]], 3
; GCN-DAG: v_mov_b32_e32 [[VAL1:v[0-9]+]], 8
; GCN-DAG: buffer_store_byte [[VAL0]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; GCN-DAG: buffer_store_dword [[VAL1]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:12

; GCN-DAG: buffer_load_dword [[RELOAD_VAL0:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:8
; GCN-DAG: buffer_load_dword [[RELOAD_VAL1:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:12

; GCN-NOT: s_add_u32 [[SP]]
; GCN-DAG: buffer_store_dword [[RELOAD_VAL0]], off, s{{\[[0-9]+:[0-9]+\]}}, [[SP]]{{$}}
; GCN-DAG: buffer_store_dword [[RELOAD_VAL1]], off, s{{\[[0-9]+:[0-9]+\]}}, [[SP]] offset:4
; GCN: s_swappc_b64
; GCN-DAG: buffer_load_ubyte [[LOAD_OUT_VAL0:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:16
; GCN-DAG: buffer_load_dword [[LOAD_OUT_VAL1:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:20
; GCN-NOT: s_sub_u32 [[SP]]

; GCN: buffer_store_byte [[LOAD_OUT_VAL0]], off
; GCN: buffer_store_dword [[LOAD_OUT_VAL1]], off
define amdgpu_kernel void @test_call_external_void_func_sret_struct_i8_i32_byval_struct_i8_i32(i32) #0 {
  %in.val = alloca { i8, i32 }, align 4, addrspace(5)
  %out.val = alloca { i8, i32 }, align 4, addrspace(5)
  %in.gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %in.val, i32 0, i32 0
  %in.gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %in.val, i32 0, i32 1
  store i8 3, i8 addrspace(5)* %in.gep0
  store i32 8, i32 addrspace(5)* %in.gep1
  call void @external_void_func_sret_struct_i8_i32_byval_struct_i8_i32({ i8, i32 } addrspace(5)* %out.val, { i8, i32 } addrspace(5)* %in.val)
  %out.gep0 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %out.val, i32 0, i32 0
  %out.gep1 = getelementptr inbounds { i8, i32 }, { i8, i32 } addrspace(5)* %out.val, i32 0, i32 1
  %out.val0 = load i8, i8 addrspace(5)* %out.gep0
  %out.val1 = load i32, i32 addrspace(5)* %out.gep1

  store volatile i8 %out.val0, i8 addrspace(1)* undef
  store volatile i32 %out.val1, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_v16i8:
define amdgpu_kernel void @test_call_external_void_func_v16i8() #0 {
  %ptr = load <16 x i8> addrspace(1)*, <16 x i8> addrspace(1)* addrspace(4)* undef
  %val = load <16 x i8>, <16 x i8> addrspace(1)* %ptr
  call void @external_void_func_v16i8(<16 x i8> %val)
  ret void
}

; GCN-LABEL: {{^}}stack_passed_arg_alignment_v32i32_f64:
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s32{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s32 offset:4
; GCN: s_swappc_b64
define amdgpu_kernel void @stack_passed_arg_alignment_v32i32_f64(<32 x i32> %val, double %tmp) #0 {
entry:
  call void @stack_passed_f64_arg(<32 x i32> %val, double %tmp)
  ret void
}

; GCN-LABEL: {{^}}tail_call_byval_align16:
; GCN-NOT: s32
; GCN: buffer_store_dword v32, off, s[0:3], s32 offset:12 ; 4-byte Folded Spill
; GCN: buffer_store_dword v33, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; GCN: buffer_load_dword v32, off, s[0:3], s32 offset:20
; GCN: buffer_load_dword v33, off, s[0:3], s32 offset:16

; GCN: s_getpc_b64

; GCN: buffer_store_dword v32, off, s[0:3], s32 offset:4
; GCN: buffer_store_dword v33, off, s[0:3], s32{{$}}
; GCN: buffer_load_dword v33, off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; GCN: buffer_load_dword v32, off, s[0:3], s32 offset:12 ; 4-byte Folded Reload
; GCN-NOT: s32
; GCN: s_setpc_b64
define void @tail_call_byval_align16(<32 x i32> %val, double %tmp) #0 {
entry:
  %alloca = alloca double, align 8, addrspace(5)
  tail call void @byval_align16_f64_arg(<32 x i32> %val, double addrspace(5)* byval align 16 %alloca)
  ret void
}

; GCN-LABEL: {{^}}tail_call_stack_passed_arg_alignment_v32i32_f64:
; GCN-NOT: s32
; GCN: buffer_store_dword v32, off, s[0:3], s32 offset:12 ; 4-byte Folded Spill
; GCN: buffer_store_dword v33, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; GCN: buffer_load_dword v32, off, s[0:3], s32 offset:4
; GCN: buffer_load_dword v33, off, s[0:3], s32{{$}}
; GCN: s_getpc_b64
; GCN: buffer_store_dword v33, off, s[0:3], s32{{$}}
; GCN: buffer_store_dword v32, off, s[0:3], s32 offset:4
; GCN: buffer_load_dword v33, off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; GCN: buffer_load_dword v32, off, s[0:3], s32 offset:12 ; 4-byte Folded Reload
; GCN-NOT: s32
; GCN: s_setpc_b64
define void @tail_call_stack_passed_arg_alignment_v32i32_f64(<32 x i32> %val, double %tmp) #0 {
entry:
  tail call void @stack_passed_f64_arg(<32 x i32> %val, double %tmp)
  ret void
}

; GCN-LABEL: {{^}}stack_12xv3i32:
; GCN: v_mov_b32_e32 [[REG12:v[0-9]+]], 12
; GCN: v_mov_b32_e32 [[REG13:v[0-9]+]], 13
; GCN: v_mov_b32_e32 [[REG14:v[0-9]+]], 14
; GCN: v_mov_b32_e32 [[REG15:v[0-9]+]], 15
; GCN: buffer_store_dword [[REG12]], {{.*$}}
; GCN: buffer_store_dword [[REG13]], {{.*}} offset:4
; GCN: buffer_store_dword [[REG14]], {{.*}} offset:8
; GCN: buffer_store_dword [[REG15]], {{.*}} offset:12
; GCN: v_mov_b32_e32 v31, 11
; GCN: s_getpc
define void @stack_12xv3i32() #0 {
entry:
  call void @external_void_func_12xv3i32(
      <3 x i32><i32 0, i32 0, i32 0>,
      <3 x i32><i32 1, i32 1, i32 1>,
      <3 x i32><i32 2, i32 2, i32 2>,
      <3 x i32><i32 3, i32 3, i32 3>,
      <3 x i32><i32 4, i32 4, i32 4>,
      <3 x i32><i32 5, i32 5, i32 5>,
      <3 x i32><i32 6, i32 6, i32 6>,
      <3 x i32><i32 7, i32 7, i32 7>,
      <3 x i32><i32 8, i32 8, i32 8>,
      <3 x i32><i32 9, i32 9, i32 9>,
      <3 x i32><i32 10, i32 11, i32 12>,
      <3 x i32><i32 13, i32 14, i32 15>)
  ret void
}

; GCN-LABEL: {{^}}stack_12xv3f32:
; GCN: v_mov_b32_e32 [[REG12:v[0-9]+]], 0x41400000
; GCN: v_mov_b32_e32 [[REG13:v[0-9]+]], 0x41500000
; GCN: v_mov_b32_e32 [[REG14:v[0-9]+]], 0x41600000
; GCN: v_mov_b32_e32 [[REG15:v[0-9]+]], 0x41700000
; GCN: buffer_store_dword [[REG12]], {{.*$}}
; GCN: buffer_store_dword [[REG13]], {{.*}} offset:4
; GCN: buffer_store_dword [[REG14]], {{.*}} offset:8
; GCN: buffer_store_dword [[REG15]], {{.*}} offset:12
; GCN: v_mov_b32_e32 v31, 0x41300000
; GCN: s_getpc
define void @stack_12xv3f32() #0 {
entry:
  call void @external_void_func_12xv3f32(
      <3 x float><float 0.0, float 0.0, float 0.0>,
      <3 x float><float 1.0, float 1.0, float 1.0>,
      <3 x float><float 2.0, float 2.0, float 2.0>,
      <3 x float><float 3.0, float 3.0, float 3.0>,
      <3 x float><float 4.0, float 4.0, float 4.0>,
      <3 x float><float 5.0, float 5.0, float 5.0>,
      <3 x float><float 6.0, float 6.0, float 6.0>,
      <3 x float><float 7.0, float 7.0, float 7.0>,
      <3 x float><float 8.0, float 8.0, float 8.0>,
      <3 x float><float 9.0, float 9.0, float 9.0>,
      <3 x float><float 10.0, float 11.0, float 12.0>,
      <3 x float><float 13.0, float 14.0, float 15.0>)
  ret void
}

; GCN-LABEL: {{^}}stack_8xv5i32:

; GCN: v_mov_b32_e32 [[REG8:v[0-9]+]], 8
; GCN: v_mov_b32_e32 [[REG9:v[0-9]+]], 9
; GCN: v_mov_b32_e32 [[REG10:v[0-9]+]], 10
; GCN: v_mov_b32_e32 [[REG11:v[0-9]+]], 11
; GCN: v_mov_b32_e32 [[REG12:v[0-9]+]], 12
; GCN: buffer_store_dword [[REG8]], {{.*$}}
; GCN: buffer_store_dword [[REG9]], {{.*}} offset:4
; GCN: buffer_store_dword [[REG10]], {{.*}} offset:8
; GCN: buffer_store_dword [[REG11]], {{.*}} offset:12
; GCN: buffer_store_dword [[REG12]], {{.*}} offset:16
; GCN: v_mov_b32_e32 [[REG13:v[0-9]+]], 13
; GCN: v_mov_b32_e32 [[REG14:v[0-9]+]], 14
; GCN: v_mov_b32_e32 [[REG15:v[0-9]+]], 15
; GCN: buffer_store_dword [[REG13]], {{.*}} offset:20
; GCN: buffer_store_dword [[REG14]], {{.*}} offset:24
; GCN: buffer_store_dword [[REG15]], {{.*}} offset:28

; GCN: v_mov_b32_e32 v31, 7
; GCN: s_getpc
define void @stack_8xv5i32() #0 {
entry:
  call void @external_void_func_8xv5i32(
      <5 x i32><i32 0, i32 0, i32 0, i32 0, i32 0>,
      <5 x i32><i32 1, i32 1, i32 1, i32 1, i32 1>,
      <5 x i32><i32 2, i32 2, i32 2, i32 2, i32 2>,
      <5 x i32><i32 3, i32 3, i32 3, i32 3, i32 3>,
      <5 x i32><i32 4, i32 4, i32 4, i32 4, i32 4>,
      <5 x i32><i32 5, i32 5, i32 5, i32 5, i32 5>,
      <5 x i32><i32 6, i32 7, i32 8, i32 9, i32 10>,
      <5 x i32><i32 11, i32 12, i32 13, i32 14, i32 15>)
  ret void
}

; GCN-LABEL: {{^}}stack_8xv5f32:
; GCN: v_mov_b32_e32 [[REG8:v[0-9]+]], 0x41000000
; GCN: v_mov_b32_e32 [[REG9:v[0-9]+]], 0x41100000
; GCN: v_mov_b32_e32 [[REG10:v[0-9]+]], 0x41200000
; GCN: v_mov_b32_e32 [[REG11:v[0-9]+]], 0x41300000
; GCN: v_mov_b32_e32 [[REG12:v[0-9]+]], 0x41400000
; GCN: buffer_store_dword [[REG8]], {{.*$}}
; GCN: buffer_store_dword [[REG9]], {{.*}} offset:4
; GCN: buffer_store_dword [[REG10]], {{.*}} offset:8
; GCN: buffer_store_dword [[REG11]], {{.*}} offset:12
; GCN: buffer_store_dword [[REG12]], {{.*}} offset:16
; GCN: v_mov_b32_e32 [[REG13:v[0-9]+]], 0x41500000
; GCN: v_mov_b32_e32 [[REG14:v[0-9]+]], 0x41600000
; GCN: v_mov_b32_e32 [[REG15:v[0-9]+]], 0x41700000
; GCN: buffer_store_dword [[REG13]], {{.*}} offset:20
; GCN: buffer_store_dword [[REG14]], {{.*}} offset:24
; GCN: buffer_store_dword [[REG15]], {{.*}} offset:28

; GCN: v_mov_b32_e32 v31, 0x40e00000
; GCN: s_getpc
define void @stack_8xv5f32() #0 {
entry:
  call void @external_void_func_8xv5f32(
      <5 x float><float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>,
      <5 x float><float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>,
      <5 x float><float 2.0, float 2.0, float 2.0, float 2.0, float 2.0>,
      <5 x float><float 3.0, float 3.0, float 3.0, float 3.0, float 3.0>,
      <5 x float><float 4.0, float 4.0, float 4.0, float 4.0, float 4.0>,
      <5 x float><float 5.0, float 5.0, float 5.0, float 5.0, float 5.0>,
      <5 x float><float 6.0, float 7.0, float 8.0, float 9.0, float 10.0>,
      <5 x float><float 11.0, float 12.0, float 13.0, float 14.0, float 15.0>)
  ret void
}

declare hidden void @byval_align16_f64_arg(<32 x i32>, double addrspace(5)* byval align 16) #0
declare hidden void @stack_passed_f64_arg(<32 x i32>, double) #0
declare hidden void @external_void_func_12xv3i32(<3 x i32>, <3 x i32>, <3 x i32>, <3 x i32>,
    <3 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, <3 x i32>, <3 x i32>) #0
declare hidden void @external_void_func_8xv5i32(<5 x i32>, <5 x i32>, <5 x i32>, <5 x i32>,
    <5 x i32>, <5 x i32>, <5 x i32>, <5 x i32>) #0
declare hidden void @external_void_func_12xv3f32(<3 x float>, <3 x float>, <3 x float>, <3 x float>,
    <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>) #0
declare hidden void @external_void_func_8xv5f32(<5 x float>, <5 x float>, <5 x float>, <5 x float>,
    <5 x float>, <5 x float>, <5 x float>, <5 x float>) #0
attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
