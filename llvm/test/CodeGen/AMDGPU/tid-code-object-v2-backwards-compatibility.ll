; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90c -mattr=-xnack --amdhsa-code-object-version=2 < %s 2>&1 | FileCheck --check-prefix=GFX90C-VALID %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90c --amdhsa-code-object-version=2 < %s 2>&1 | FileCheck --check-prefix=GFX90C-ERROR %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 --amdhsa-code-object-version=2 < %s 2>&1 | FileCheck --check-prefix=GFX940-ERROR %s

; GFX90C-VALID: .hsa_code_object_isa 9,0,12,"AMD","AMDGPU"
; GFX90C-VALID: .amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx90c"
; GFX90C-ERROR: LLVM ERROR: AMD GPU code object V2 does not support processor gfx90c with XNACK being ON or ANY

; GFX940-ERROR: LLVM ERROR: AMD GPU code object V2 does not support processor gfx940

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}
