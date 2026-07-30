; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.1-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.1-amd-amdhsa < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.3-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.3-amd-amdhsa < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11-amd-amdhsa < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11.7-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11.7-amd-amdhsa < %s | FileCheck -check-prefix=CU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu12-amd-amdhsa -mattr=+cumode < %s | FileCheck -check-prefix=NOCU %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu12-amd-amdhsa < %s | FileCheck -check-prefix=CU %s

; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.1-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.1-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.3-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu10.3-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11.7-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu11.7-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu12-amd-amdhsa -mattr=+wavefrontsize32 < %s | FileCheck -check-prefix=W32 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu12-amd-amdhsa -mattr=+wavefrontsize64 < %s | FileCheck -check-prefix=W64 %s
; RUN: llc --amdhsa-code-object-version=6 -mtriple=amdgpu12.5-amd-amdhsa < %s | FileCheck -check-prefix=W32 %s

; Checks 10.1, 10.3, 11, 11.7 and 12 generic targets allow cumode/wave64.

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
