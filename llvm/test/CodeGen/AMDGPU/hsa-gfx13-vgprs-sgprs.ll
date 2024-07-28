; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1300 | FileCheck --check-prefix=ASM %s
; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1300 -filetype=obj | llvm-objdump -s -j .rodata - | FileCheck --check-prefix=OBJ %s

; ASM: .amdhsa_next_free_vgpr 1024
; ASM: .amdhsa_next_free_sgpr 104

; OBJ: 0000 00000000 00000000 00000000 00000000
; OBJ: 0010 00000000 00000000 00000000 00000000
; OBJ: 0020 00000000 00000000 00000000 00000000
; OBJ: 0030 7f000f40 80000000 00040000 00000000

define amdgpu_kernel void @simple() #0 {
entry:
  call void asm sideeffect "; clobber $0", "~{v1023}"()
  call void asm sideeffect "; clobber $0", "~{s103}"()
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="32,32" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
