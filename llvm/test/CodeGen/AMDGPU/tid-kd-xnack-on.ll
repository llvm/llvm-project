; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+xnack < %s | FileCheck --check-prefixes=ASM %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+xnack --filetype=obj < %s | llvm-objdump -s -j .rodata - | FileCheck --check-prefixes=OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+xnack --filetype=obj < %s | llvm-readelf --notes - | FileCheck --check-prefixes=ELF %s

; TODO: Update to check for granulated sgpr count directive once one is added.

define amdgpu_kernel void @kern() #0 {
; ASM-LABEL: kern:
; ASM: .amdhsa_next_free_sgpr 5
; ASM: .amdhsa_reserve_xnack_mask 1

; Verify that an extra SGPR block is reserved with XNACK "on" tid setting.
; OBJ: Contents of section .rodata:
; OBJ-NEXT: 0000 00000000 00000000 00000000 00000000  ................
; OBJ-NEXT: 0010 00000000 00000000 00000000 00000000  ................
; OBJ-NEXT: 0020 00000000 00000000 00000000 00000000  ................
; OBJ-NEXT: 0030 4000af00 8c000000 21000000 00000000 @.......!.......

; ELF: AMDGPU Metadata
; ELF: .sgpr_count:     9
entry:
  tail call void asm sideeffect "", "~{s[0:4]}"()
  ret void
}

attributes #0 = { "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
