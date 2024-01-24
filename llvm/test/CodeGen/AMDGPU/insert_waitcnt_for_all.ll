; Testing the -amdgpu-precise-memory-op option
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX9
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX90A
; COM: llc -mtriple=amdgcn -mcpu=gfx1010 -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX10
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx900 -mattr=-flat-for-global,+enable-flat-scratch,+amdgpu-precise-memory-op -amdgpu-use-divergent-register-indexing -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX9-FLATSCR %s

; from atomicrmw-expand.ll
; covers flat_load, flat_atomic
define void @syncscope_workgroup_nortn(ptr %addr, float %val) {
; GFX90A-LABEL: syncscope_workgroup_nortn:
; GFX90A:  ; %bb.0:
; GFX90A:         flat_load_dword v5, v[0:1]
; GFX90A-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX90A:  .LBB0_1: ; %atomicrmw.start
; GFX90A:         flat_atomic_cmpswap v3, v[0:1], v[4:5] glc
; GFX90A-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
  %res = atomicrmw fadd ptr %addr, float %val syncscope("workgroup") seq_cst
  ret void
}

; from atomicrmw-nand.ll
; covers global_atomic, global_load
define i32 @atomic_nand_i32_global(ptr addrspace(1) %ptr) nounwind {
; GFX9-LABEL: atomic_nand_i32_global:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_dword v2, v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_mov_b64 s[4:5], 0
; GFX9-NEXT:  .LBB1_1: ; %atomicrmw.start
; GFX9-NEXT:    ; =>This Inner Loop Header: Depth=1
; GFX9-NOT:     s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v3, v2
; GFX9-NEXT:    v_not_b32_e32 v2, v3
; GFX9-NEXT:    v_or_b32_e32 v2, -5, v2
; GFX9-NEXT:    global_atomic_cmpswap v2, v[0:1], v[2:3], off glc
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX9-NEXT:    buffer_wbinvl1_vol
; GFX9-NEXT:    v_cmp_eq_u32_e32 vcc, v2, v3
; GFX9-NEXT:    s_or_b64 s[4:5], vcc, s[4:5]
; GFX9-NEXT:    s_andn2_b64 exec, exec, s[4:5]
; GFX9-NEXT:    s_cbranch_execnz .LBB1_1
; GFX9-NEXT:  ; %bb.2: ; %atomicrmw.end
; GFX9-NEXT:    s_or_b64 exec, exec, s[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v0, v2
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = atomicrmw nand ptr addrspace(1) %ptr, i32 4 seq_cst
  ret i32 %result
}

; from bf16.ll
; covers buffer_load, buffer_store, flat_load, flat_store, global_load, global_store
define void @test_load_store(ptr addrspace(1) %in, ptr addrspace(1) %out) {
;
; GFX9-LABEL: test_load_store:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_ushort v0, v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_store_short v[2:3], v0, off
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
;
; GFX10-LABEL: test_load_store:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    global_load_ushort v0, v[0:1], off
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10-NEXT:    global_store_short v[2:3], v0, off
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:    s_waitcnt_vscnt null, 0x0
; GFX10-NEXT:    s_setpc_b64 s[30:31]
  %val = load bfloat, ptr addrspace(1) %in
  store bfloat %val, ptr addrspace(1) %out
  ret void
}

; from scratch-simple.ll
; covers scratch_load, scratch_store
;
; GFX9-FLATSCR-LABEL: {{^}}vs_main:
; GFX9-FLATSCR:        scratch_store_dwordx4 off, v[{{[0-9:]+}}],
; GFX9-FLATSCR-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX9-FLATSCR:        scratch_load_dword {{v[0-9]+}}, {{v[0-9]+}}, off
; GFX9-FLATSCR-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0)
define amdgpu_vs float @vs_main(i32 %idx) {
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}

; from udiv.ll
; covers s_load
define amdgpu_kernel void @udiv_i32(ptr addrspace(1) %out, i32 %x, i32 %y) {
; GFX9-LABEL: udiv_i32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v1, 0
; GFX9-NOT:     s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_cvt_f32_u32_e32 v0, s3
  %r = udiv i32 %x, %y
  store i32 %r, ptr addrspace(1) %out
  ret void
}

declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32)

; from smrd.ll
; covers s_buffer_load
; GFX9-LABEL: {{^}}smrd_sgpr_offset:
; GFX9:         s_buffer_load_dword s{{[0-9]}}, s[0:3], s4
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
define amdgpu_ps float @smrd_sgpr_offset(<4 x i32> inreg %desc, i32 inreg %offset) #0 {
main_body:
  %r = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %desc, i32 %offset, i32 0)
  ret float %r
}

; from atomic_load_add.ll
; covers s_load, ds_add
; GFX9-LABEL: atomic_add_local:
; GFX9:       ; %bb.1:
; GFX9-NEXT:    s_load_dword s0, s[0:1], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9:         ds_add_u32 v0, v1
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
define amdgpu_kernel void @atomic_add_local(ptr addrspace(3) %local) {
   %unused = atomicrmw volatile add ptr addrspace(3) %local, i32 5 seq_cst
   ret void
}

declare i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add(i32, ptr addrspace(8), i32, i32, i32 immarg)

; from atomic_optimizations_buffer.ll
; covers buffer_atomic
; GFX9-LABEL: add_i32_constant:
; GFX9:       ; %bb.1:
; GFX9-NEXT:    s_load_dwordx4 s[8:11], s[0:1], 0x34
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9:         buffer_atomic_add v1, off, s[8:11], 0 glc
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
define amdgpu_kernel void @add_i32_constant(ptr addrspace(1) %out, ptr addrspace(8) %inout) {
entry:
  %old = call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add(i32 5, ptr addrspace(8) %inout, i32 0, i32 0, i32 0)
  store i32 %old, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32, i16, <8 x i32>, i32, i32)

; from llvm.amdgcn.image.load.a16.ll
; covers image_load
; GFX9-LABEL: {{^}}load.f32.1d:
; GFX9:         image_load v0, v0, s[0:7] dmask:0x1 unorm a16
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
define amdgpu_ps <4 x float> @load.f32.1d(<8 x i32> inreg %rsrc, <2 x i16> %coords) {
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  %v = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i16(i32 1, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

declare void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float>, i32, i16, <8 x i32>, i32, i32)

; from llvm.amdgcn.image.store.a16.ll
; covers image_store
define amdgpu_ps void @store_f32_1d(<8 x i32> inreg %rsrc, <2 x i16> %coords, <4 x float> %val) {
; GFX9-LABEL: store_f32_1d:
; GFX9:       ; %bb.0: ; %main_body
; GFX9-NEXT:    image_store v[1:4], v0, s[0:7] dmask:0x1 unorm a16
; GFX9-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_endpgm
;
main_body:
  %x = extractelement <2 x i16> %coords, i32 0
  call void @llvm.amdgcn.image.store.1d.v4f32.i16(<4 x float> %val, i32 1, i16 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

declare i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32, i32, <8 x i32>, i32, i32)

; from llvm.amdgcn.image.atomic.dim.ll
; covers image_atomic
; GFX90A-LABEL: {{^}}atomic_swap_1d:
; GFX90A: image_atomic_swap v0, v{{[02468]}}, s[0:7] dmask:0x1 unorm glc{{$}}
; GFX90A-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
define amdgpu_ps float @atomic_swap_1d(<8 x i32> inreg %rsrc, i32 %data, i32 %s) {
main_body:
  %v = call i32 @llvm.amdgcn.image.atomic.swap.1d.i32.i32(i32 %data, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  %out = bitcast i32 %v to float
  ret float %out
}




