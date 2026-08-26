; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; A module-scope global emits OpUntypedVariableKHR at module scope with the
; global's value type.

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#CROSS:]] = OpTypeUntypedPointerKHR CrossWorkgroup

; The variable is emitted before the first OpFunction.
; CHECK: OpUntypedVariableKHR %[[#CROSS]] CrossWorkgroup %[[#I32]]
; CHECK: OpFunction

@g = addrspace(1) global i32 42, align 4
define spir_kernel void @use(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(1) @g, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
