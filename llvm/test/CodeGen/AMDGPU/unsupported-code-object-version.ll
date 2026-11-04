; RUN: sed 's/CODE_OBJECT_VERSION/0/g' %s | not llc -mtriple=amdgpu9.06-amd-amdhsa 2>&1 | FileCheck --check-prefix=HSA-ERROR %s
; RUN: sed 's/CODE_OBJECT_VERSION/100/g' %s | not llc -mtriple=amdgpu9.06-amd-amdhsa 2>&1 | FileCheck --check-prefix=HSA-ERROR %s
; RUN: sed 's/CODE_OBJECT_VERSION/9900/g' %s | not llc -mtriple=amdgpu9.06-amd-amdhsa 2>&1 | FileCheck --check-prefix=HSA-ERROR %s
; RUN: sed 's/CODE_OBJECT_VERSION/0/g' %s | not llc -filetype=obj -mtriple=amdgpu9.06-amd-amdhsa 2>&1 | FileCheck --check-prefix=HSA-ERROR %s
; RUN: sed 's/CODE_OBJECT_VERSION/0/g' %s | not llc -filetype=asm -mtriple=amdgpu9.06-amd-amdhsa 2>&1 | FileCheck --check-prefix=HSA-ERROR %s

; HSA-ERROR: unsupported code object version

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 CODE_OBJECT_VERSION}
