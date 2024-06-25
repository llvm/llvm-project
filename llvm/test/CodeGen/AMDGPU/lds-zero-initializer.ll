; RUN: llc -mtriple=amdgcn -mcpu=tahiti -stop-after=amdgpu-isel -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GCN,GFX6 %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -stop-after=amdgpu-isel  -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GCN,GFX8 %s

; RUN: not llc -mtriple=amdgcn -mcpu=tahiti < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn -mcpu=tonga  < %s 2>&1 | FileCheck %s

; CHECK: error: lds: unsupported initializer for address space

@lds = addrspace(3) global [256 x i32] zeroinitializer

define amdgpu_kernel void @load_zeroinit_lds_global(ptr addrspace(1) %out, i1 %p) {
  ; GCN-LABEL: name: load_zeroinit_lds_global
  ; GCN: bb.0 (%ir-block.0):
  ; GCN:   liveins: $sgpr0_sgpr1
  ; GCN:   [[COPY:%[0-9]+]]:sgpr_64(p4) = COPY $sgpr0_sgpr1
  ; GFX6:  [[S_LOAD_DWORDX2_IMM:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM [[COPY]](p4), 9, 0
  ; GFX8:  [[S_LOAD_DWORDX2_IMM:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM [[COPY]](p4), 36, 0
  ; GFX6:  [[COPY1:%[0-9]+]]:sreg_32 = COPY [[S_LOAD_DWORDX2_IMM]].sub1
  ; GFX6:  [[COPY2:%[0-9]+]]:sreg_32 = COPY [[S_LOAD_DWORDX2_IMM]].sub0
  ; GFX6:  [[S_MOV_B32_:%[0-9]+]]:sreg_32 = S_MOV_B32 61440
  ; GFX6:  [[S_MOV_B32_1:%[0-9]+]]:sreg_32 = S_MOV_B32 -1
  ; GFX6:  [[REG_SEQUENCE:%[0-9]+]]:sgpr_128 = REG_SEQUENCE killed [[COPY2]], %subreg.sub0, killed [[COPY1]], %subreg.sub1, killed [[S_MOV_B32_1]], %subreg.sub2, killed [[S_MOV_B32_]], %subreg.sub3
  ; GCN:   [[V_MOV_B32_e32_:%[0-9]+]]:vgpr_32 = V_MOV_B32_e32 target-flags(amdgpu-abs32-lo) @lds, implicit $exec
  ; GCN:   SI_INIT_M0 -1, implicit-def $m0
  ; GCN:   [[DS_READ_B32_:%[0-9]+]]:vgpr_32 = DS_READ_B32 killed [[V_MOV_B32_e32_]], 40, 0, implicit $m0, implicit $exec
  ; GFX8:  [[COPY1:%[0-9]+]]:vreg_64 = COPY [[S_LOAD_DWORDX2_IMM]]
  ; GFX6:  BUFFER_STORE_DWORD_OFFSET killed [[DS_READ_B32_]], killed [[REG_SEQUENCE]], 0, 0, 0, 0, implicit $exec
  ; GFX8:  FLAT_STORE_DWORD killed [[COPY1]], killed [[DS_READ_B32_]], 0, 0, implicit $exec, implicit $flat_scr
  ; GCN:   S_ENDPGM 0
 %gep = getelementptr [256 x i32], ptr addrspace(3) @lds, i32 0, i32 10
  %ld = load i32, ptr addrspace(3) %gep
  store i32 %ld, ptr addrspace(1) %out
  ret void
}
