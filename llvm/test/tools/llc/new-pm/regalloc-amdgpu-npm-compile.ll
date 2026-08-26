; REQUIRES: amdgpu-registered-target
; RUN: llc -verify-machineinstrs -enable-new-pm -O3 -sgpr-regalloc-npm=basic -wwm-regalloc-npm=basic -vgpr-regalloc-npm=basic -mtriple=amdgpu7.00-amd-amdhsa -filetype=null %s

declare void @bar()

; Exercise SGPR allocation with basic NPM register allocators.
define void @foo() {
  call void asm sideeffect "; clobber", "~{s33}"()
  call void @bar()
  ret void
}
