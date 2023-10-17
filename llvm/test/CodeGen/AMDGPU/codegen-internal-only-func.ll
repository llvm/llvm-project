; REQUIRES: asserts
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=OPT %s
; RUN: not llc --crash -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s

; AMDGPUAttributor deletes the function "by accident" so it's never
; codegened with optimizations.

; OPT:	  .text
; OPT-NEXT: .section	".note.GNU-stack"
; OPT-NEXT: .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
; OPT-NEXT: .amdgpu_metadata
; OPT-NEXT: ---
; OPT-NEXT: amdhsa.kernels:  []
; OPT-NEXT: amdhsa.target:   amdgcn-amd-amdhsa--gfx900
; OPT-NEXT: amdhsa.version:
; OPT-NEXT: - 1
; OPT-NEXT: - 1
; OPT-NEXT: ...
define internal i32 @func() {
  ret i32 0
}

