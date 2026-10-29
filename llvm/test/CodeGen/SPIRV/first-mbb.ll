; Test the case where we create a type in the entry block without having set it before.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability Addresses
; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability AsmINTEL

define spir_func void @foo(i32 noundef %v) {
entry:
  call addrspace(0) void asm sideeffect "s_mov_b32 m0, $0", "s,~{memory}"(i32 %v)
  ret void
}