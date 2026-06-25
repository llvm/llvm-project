; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 < %s 2>&1 | FileCheck %s

; The "VGPR as memory" file is a block of reserved physical VGPRs. Inline asm
; that explicitly clobbers one of those registers would corrupt the file, so
; AMDGPUPrivateObjectVGPRs diagnoses it after register allocation, where the
; reserved registers are final. (For this function the file is at v2.)

@g = internal addrspace(13) global i32 poison

; CHECK: error: {{.*}}inline asm clobbers a 'VGPR as memory' reserved register
define void @asm_clobber(i32 %v) {
  store i32 %v, ptr addrspace(13) @g
  call void asm sideeffect "", "~{v2}"()
  ret void
}
