; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-1-generic -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-1-generic < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-3-generic -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-3-generic < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx11-generic -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx11-generic < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx12-generic < %s | FileCheck -check-prefix=CU %s

; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-1-generic -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-1-generic -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-3-generic -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx10-3-generic -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx11-generic -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx11-generic -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx12-generic -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s

; Checks 10.1, 10.3 and 11 generic targets allow cumode/wave64.

; NOCU:    .amdhsa_workgroup_processor_mode 0
; NOCU:    .workgroup_processor_mode: 0
; CU:      .amdhsa_workgroup_processor_mode 1
; CU:      .workgroup_processor_mode: 1

; W64:      .amdhsa_wavefront_size32 0
; W32:      .amdhsa_wavefront_size32 1

define amdgpu_kernel void @wavefrontsize() {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
