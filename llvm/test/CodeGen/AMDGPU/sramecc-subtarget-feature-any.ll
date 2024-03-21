; RUN: llc -mtriple=amdgcn -mcpu=gfx700 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=NOT-SUPPORTED %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -debug-only=amdgpu-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ANY %s

; REQUIRES: asserts

; NOT-SUPPORTED: sramecc setting for subtarget: Unsupported
; ANY: sramecc setting for subtarget: Any
define void @sramecc-subtarget-feature-default() nounwind {
  ret void
}
