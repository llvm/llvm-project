; REQUIRES: amdgpu-registered-target, x86-registered-target
; NOTE: Do not autogenerate. Check the visibility of the shared profile version.
; RUN: opt -mtriple=amdgpu9.50-amd-amdhsa -passes=pgo-instr-gen -S < %s | FileCheck %s --check-prefix=GPU
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=pgo-instr-gen -S < %s | FileCheck %s --check-prefix=CPU

; GPU: @__llvm_profile_raw_version = protected addrspace(1) constant i64
; CPU: @__llvm_profile_raw_version = hidden constant i64

define void @profiled() {
entry:
  ret void
}
