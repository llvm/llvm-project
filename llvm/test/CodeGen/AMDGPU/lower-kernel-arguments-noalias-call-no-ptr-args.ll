; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=amdgpu-lower-kernel-arguments %s | FileCheck %s

; Regression test for a bug where addAliasScopeMetadata skipped memory-
; accessing calls with no pointer arguments, leaving them without !noalias
; metadata. This caused AA to conservatively report them as potential
; clobbers of noalias kernel arguments, blocking downstream scalarization
; in AMDGPUAnnotateUniformValues and causing severe performance regressions
; (e.g. in rocFFT).

declare i32 @memory_read_no_ptr_args() #1

; The call reads memory but has no pointer arguments — it cannot alias
; any noalias kernel argument. The pass must add !noalias metadata to it.
define amdgpu_kernel void @call_without_ptr_args(ptr addrspace(1) noalias %out, ptr addrspace(1) noalias %in) #0 {
; CHECK-LABEL: @call_without_ptr_args(
; CHECK: call i32 @memory_read_no_ptr_args(), !noalias [[SCOPES:![0-9]+]]
; CHECK: load i32, {{.*}} !alias.scope {{.*}} !noalias
; CHECK: store i32 {{.*}} !alias.scope {{.*}} !noalias
; CHECK: ret void
  %val = call i32 @memory_read_no_ptr_args()
  %gep = getelementptr i32, ptr addrspace(1) %in, i32 %val
  %load = load i32, ptr addrspace(1) %gep, align 4
  store i32 %load, ptr addrspace(1) %out, align 4
  ret void
}

; Same scenario but the call is readnone — should NOT get noalias metadata
; because it doesn't access memory at all and is skipped by the pass.
declare i32 @readnone_no_ptr_args() #2

define amdgpu_kernel void @readnone_call_without_ptr_args(ptr addrspace(1) noalias %out) #0 {
; CHECK-LABEL: @readnone_call_without_ptr_args(
; CHECK: {{call i32 @readnone_no_ptr_args\(\)$}}
; CHECK: store i32
  %val = call i32 @readnone_no_ptr_args()
  store i32 %val, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind memory(read) }
attributes #2 = { nounwind memory(none) }
