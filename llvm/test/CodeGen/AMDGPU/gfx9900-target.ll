; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9900 -verify-machineinstrs < %s | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=amdgpu99.00-amd-amdhsa -verify-machineinstrs < %s | FileCheck %s --check-prefix=TRIPLE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx9900 -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck %s --check-prefix=ELF

; LEGACY: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx9900"
; TRIPLE: .amdgcn_target "amdgpu99.00-amd-amdhsa-unknown-gfx9900"
; ELF: EF_AMDGPU_MACH_AMDGCN_GFX9900

define amdgpu_kernel void @gfx9900_kernel() {
  ret void
}
