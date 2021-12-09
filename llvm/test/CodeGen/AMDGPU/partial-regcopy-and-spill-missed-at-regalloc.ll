;RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 --stop-after=greedy,1 -verify-machineinstrs < %s | FileCheck -check-prefix=REGALLOC-GFX908 %s
;RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 --stop-after=prologepilog -verify-machineinstrs < %s | FileCheck -check-prefix=PEI-GFX908 %s
;RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a --stop-after=greedy,1 -verify-machineinstrs < %s | FileCheck -check-prefix=REGALLOC-GFX90A %s
;RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a --stop-after=prologepilog -verify-machineinstrs < %s | FileCheck -check-prefix=PEI-GFX90A %s

; Partial reg copy and spill missed during regalloc handled later at frame lowering.
define amdgpu_kernel void @partial_copy(<4 x i32> %arg) #0 {
  ; REGALLOC-GFX908-LABEL: name: partial_copy
  ; REGALLOC-GFX908: bb.0 (%ir-block.0):
  ; REGALLOC-GFX908:  INLINEASM &"; def $0", 1 /* sideeffect attdialect */, 2949130 /* regdef:VReg_64 */, def [[VREG_64:%[0-9]+]]
  ; REGALLOC-GFX908:  SI_SPILL_V64_SAVE [[VREG_64]], %stack.0
  ; REGALLOC-GFX908:  [[V_MFMA_I32_4X4X4I8_A128:%[0-9]+]]:areg_128 = V_MFMA_I32_4X4X4I8_e64
  ; REGALLOC-GFX908:  [[SI_SPILL_V64_RESTORE:%[0-9]+]]:vreg_64 = SI_SPILL_V64_RESTORE %stack.0
  ; REGALLOC-GFX908:  GLOBAL_STORE_DWORDX2 undef %{{[0-9]+}}:vreg_64, [[SI_SPILL_V64_RESTORE]]
  ; REGALLOC-GFX908:  [[COPY_A128_TO_V128:%[0-9]+]]:vreg_128 = COPY [[V_MFMA_I32_4X4X4I8_A128]]
  ; REGALLOC-GFX908:  GLOBAL_STORE_DWORDX4 undef %{{[0-9]+}}:vreg_64, [[COPY_A128_TO_V128]]
  ;
  ; PEI-GFX908-LABEL: name: partial_copy
  ; PEI-GFX908: bb.0 (%ir-block.0):
  ; PEI-GFX908:  INLINEASM &"; def $0", 1 /* sideeffect attdialect */, 2949130 /* regdef:VReg_64 */, def renamable $vgpr0_vgpr1
  ; PEI-GFX908:  BUFFER_STORE_DWORD_OFFSET killed $vgpr0
  ; PEI-GFX908:  $agpr4 = V_ACCVGPR_WRITE_B32_e64 killed $vgpr1
  ; PEI-GFX908:  renamable $agpr0_agpr1_agpr2_agpr3 = V_MFMA_I32_4X4X4I8_e64
  ; PEI-GFX908:  $vgpr0 = BUFFER_LOAD_DWORD_OFFSET
  ; PEI-GFX908:  $vgpr1 = V_ACCVGPR_READ_B32_e64 $agpr4
  ; PEI-GFX908:  GLOBAL_STORE_DWORDX2 undef renamable ${{.*}}, killed renamable $vgpr0_vgpr1
  ; PEI-GFX908:  renamable $vgpr0_vgpr1_vgpr2_vgpr3 = COPY killed renamable $agpr0_agpr1_agpr2_agpr3, implicit $exec
  ; PEI-GFX908:  GLOBAL_STORE_DWORDX4 undef renamable ${{.*}}, killed renamable $vgpr0_vgpr1_vgpr2_vgpr3
  ;
  ; REGALLOC-GFX90A-LABEL: name: partial_copy
  ; REGALLOC-GFX90A: bb.0 (%ir-block.0):
  ; REGALLOC-GFX90A:  INLINEASM &"; def $0", 1 /* sideeffect attdialect */, 3080202 /* regdef:VReg_64_Align2 */, def [[VREG_64:%[0-9]+]]
  ; REGALLOC-GFX90A:  SI_SPILL_V64_SAVE [[VREG_64]], %stack.0
  ; REGALLOC-GFX90A:  [[V_MFMA_I32_4X4X4I8_A128:%[0-9]+]]:areg_128_align2 = V_MFMA_I32_4X4X4I8_e64
  ; REGALLOC-GFX90A:  [[SI_SPILL_AV64_RESTORE:%[0-9]+]]:av_64_align2 = SI_SPILL_AV64_RESTORE %stack.0
  ; REGALLOC-GFX90A:  GLOBAL_STORE_DWORDX2 undef %{{[0-9]+}}:vreg_64_align2, [[SI_SPILL_AV64_RESTORE]]
  ; REGALLOC-GFX90A:  GLOBAL_STORE_DWORDX4 undef %{{[0-9]+}}:vreg_64_align2, [[V_MFMA_I32_4X4X4I8_A128]]
  ;
  ; PEI-GFX90A-LABEL: name: partial_copy
  ; PEI-GFX90A: bb.0 (%ir-block.0):
  ; PEI-GFX90A:  INLINEASM &"; def $0", 1 /* sideeffect attdialect */, 3080202 /* regdef:VReg_64_Align2 */, def renamable $vgpr0_vgpr1
  ; PEI-GFX90A:  BUFFER_STORE_DWORD_OFFSET killed $vgpr0
  ; PEI-GFX90A:  $agpr4 = V_ACCVGPR_WRITE_B32_e64 killed $vgpr1
  ; PEI-GFX90A:  renamable $agpr0_agpr1_agpr2_agpr3 = V_MFMA_I32_4X4X4I8_e64
  ; PEI-GFX90A:  $vgpr0 = BUFFER_LOAD_DWORD_OFFSET
  ; PEI-GFX90A:  $vgpr1 = V_ACCVGPR_READ_B32_e64 $agpr4
  ; PEI-GFX90A:  GLOBAL_STORE_DWORDX2 undef renamable ${{.*}}, killed renamable $vgpr0_vgpr1
  ; PEI-GFX90A:  GLOBAL_STORE_DWORDX4 undef renamable ${{.*}}, killed renamable $agpr0_agpr1_agpr2_agpr3
  %v0 = call <4 x i32> asm sideeffect "; def $0", "=v" ()
  %v1 = call <2 x i32> asm sideeffect "; def $0", "=v" ()
  %mai = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %arg, i32 0, i32 0, i32 0)
  store volatile <4 x i32> %v0, <4 x i32> addrspace(1)* undef
  store volatile <2 x i32> %v1, <2 x i32> addrspace(1)* undef
  store volatile <4 x i32> %mai, <4 x i32> addrspace(1)* undef
  ret void
}

declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)

attributes #0 = { nounwind "amdgpu-num-vgpr"="5" }
