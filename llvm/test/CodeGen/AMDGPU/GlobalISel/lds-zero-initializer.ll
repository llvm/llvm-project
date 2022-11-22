; RUN: llc -march=amdgcn -mcpu=tahiti -global-isel -stop-after=instruction-select -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -march=amdgcn -mcpu=tonga -global-isel -stop-after=instruction-select -verify-machineinstrs -o - %s | FileCheck -check-prefixes=GCN,GFX9 %s

; RUN: not llc -march=amdgcn -mcpu=tahiti -global-isel < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mcpu=tonga -global-isel < %s 2>&1 | FileCheck %s

; CHECK: error: lds: unsupported initializer for address space

@lds = addrspace(3) global [256 x i32] zeroinitializer

define amdgpu_kernel void @load_zeroinit_lds_global(i32 addrspace(1)* %out, i1 %p) {
  ; GCN-LABEL: name: load_zeroinit_lds_global
  ; GCN: bb.1 (%ir-block.0):
  ; GCN:   liveins: $sgpr0_sgpr1
  ; GCN:   [[COPY:%[0-9]+]]:sreg_64 = COPY $sgpr0_sgpr1
  ; GFX8:  [[S_MOV_B32_:%[0-9]+]]:sreg_32 = S_MOV_B32 40
  ; GCN:   [[S_MOV_B32_1:%[0-9]+]]:sreg_32 = S_MOV_B32 target-flags(amdgpu-abs32-lo) @lds
  ; GFX8:  [[S_ADD_U32_:%[0-9]+]]:sreg_32 = S_ADD_U32 [[S_MOV_B32_1]], [[S_MOV_B32_]], implicit-def $scc
  ; GFX8:  [[S_LOAD_DWORDX2_IMM:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM [[COPY]], 9, 0
  ; GFX9:  [[S_LOAD_DWORDX2_IMM:%[0-9]+]]:sreg_64_xexec = S_LOAD_DWORDX2_IMM [[COPY]], 36, 0
  ; GFX8:  [[COPY1:%[0-9]+]]:vgpr_32 = COPY [[S_ADD_U32_]]
  ; GCN:   $m0 = S_MOV_B32 -1
  ; GFX9:  [[COPY1:%[0-9]+]]:vgpr_32 = COPY [[S_MOV_B32_1]]
  ; GFX8:  [[DS_READ_B32_:%[0-9]+]]:vgpr_32 = DS_READ_B32 [[COPY1]], 0, 0, implicit $m0, implicit $exec
  ; GFX9:  [[DS_READ_B32_:%[0-9]+]]:vgpr_32 = DS_READ_B32 [[COPY1]], 40, 0, implicit $m0, implicit $exec
  ; GFX8:  [[S_MOV_B32_2:%[0-9]+]]:sreg_32 = S_MOV_B32 4294967295
  ; GFX8:  [[S_MOV_B32_3:%[0-9]+]]:sreg_32 = S_MOV_B32 61440
  ; GFX8:  [[REG_SEQUENCE:%[0-9]+]]:sreg_64 = REG_SEQUENCE [[S_MOV_B32_2]], %subreg.sub0, [[S_MOV_B32_3]], %subreg.sub1
  ; GFX8:  [[REG_SEQUENCE1:%[0-9]+]]:sgpr_128 = REG_SEQUENCE [[S_LOAD_DWORDX2_IMM]], %subreg.sub0_sub1, [[REG_SEQUENCE]], %subreg.sub2_sub3
  ; GFX8:  BUFFER_STORE_DWORD_OFFSET [[DS_READ_B32_]], [[REG_SEQUENCE1]], 0, 0, 0, 0, implicit $exec
  ; GFX9:  [[COPY2:%[0-9]+]]:vreg_64 = COPY [[S_LOAD_DWORDX2_IMM]]
  ; GFX9:  FLAT_STORE_DWORD [[COPY2]], [[DS_READ_B32_]], 0, 0, implicit $exec, implicit $flat_scr
  ; GCN:   S_ENDPGM 0
 %gep = getelementptr [256 x i32], [256 x i32] addrspace(3)* @lds, i32 0, i32 10
  %ld = load i32, i32 addrspace(3)* %gep
  store i32 %ld, i32 addrspace(1)* %out
  ret void
}
