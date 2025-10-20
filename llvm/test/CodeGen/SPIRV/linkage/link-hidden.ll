; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[BAR:[0-9]+]] "bar"
; CHECK: OpDecorate %[[BAR]] LinkageAttributes "bar" Import
; CHECK: %[[BAR]] = OpFunction

define hidden spir_kernel void @foo() addrspace(4) {
entry:
  call spir_func addrspace(4) void @bar()
  ret void
}

declare hidden spir_func void @bar() addrspace(4)
