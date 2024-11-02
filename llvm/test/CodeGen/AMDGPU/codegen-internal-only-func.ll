; REQUIRES: asserts
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=OPT,COV4 %s
; RUN: not llc --crash -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s
; RUN: sed 's/CODE_OBJECT_VERSION/500/g' %s | llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 | FileCheck -check-prefixes=OPT,COV5 %s

; AMDGPUAttributor deletes the function "by accident" so it's never
; codegened with optimizations.

; OPT:	  .text
; OPT-NEXT: .section	".note.GNU-stack"
; OPT-NEXT: .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
; COV4-NEXT: .amdhsa_code_object_version 4
; COV5-NEXT: .amdhsa_code_object_version 5
; OPT-NEXT: .amdgpu_metadata
; OPT-NEXT: ---
; OPT-NEXT: amdhsa.kernels:  []
; OPT-NEXT: amdhsa.target:   amdgcn-amd-amdhsa--gfx900
; OPT-NEXT: amdhsa.version:
; OPT-NEXT: - 1
; COV4: - 1
; COV5: - 2
; OPT: ...
define internal i32 @func() {
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 CODE_OBJECT_VERSION}
