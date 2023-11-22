; Testing the -amdgpu-precise-memory-op option
; RUN: llc -mtriple=amdgcn -mcpu=hawaii -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX7
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX8
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX9
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -mattr=+amdgpu-precise-memory-op -verify-machineinstrs < %s | FileCheck %s -check-prefixes=GFX10

; from bf16.ll
; covers buffer_load, buffer_store, flat_load, flat_store, global_load, global_store
define void @test_load_store(ptr addrspace(1) %in, ptr addrspace(1) %out) {
;
; GFX7-LABEL: test_load_store:
; GFX7:       ; %bb.0:
; GFX7-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7-NEXT:    s_mov_b32 s6, 0
; GFX7-NEXT:    s_mov_b32 s7, 0xf000
; GFX7-NEXT:    s_mov_b32 s4, s6
; GFX7-NEXT:    s_mov_b32 s5, s6
; GFX7-NEXT:    buffer_load_ushort v0, v[0:1], s[4:7], 0 addr64
; GFX7-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7-NEXT:    buffer_store_short v0, v[2:3], s[4:7], 0 addr64
; GFX7-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX7-NEXT:    s_setpc_b64 s[30:31]
;
; GFX8-LABEL: test_load_store:
; GFX8:       ; %bb.0:
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    flat_load_ushort v0, v[0:1]
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    flat_store_short v[2:3], v0
; GFX8-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX8-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: test_load_store:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_load_ushort v0, v[0:1], off
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    global_store_short v[2:3], v0, off
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
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

