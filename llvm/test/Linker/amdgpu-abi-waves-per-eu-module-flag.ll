; RUN: split-file %s %t
; RUN: not llvm-link %t/a.ll %t/b.ll -S -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: linking module flags 'amdgpu_abi_waves_per_eu': IDs have conflicting values

;--- a.ll
target triple = "amdgcn-amd-amdhsa"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}

;--- b.ll
target triple = "amdgcn-amd-amdhsa"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 4}
