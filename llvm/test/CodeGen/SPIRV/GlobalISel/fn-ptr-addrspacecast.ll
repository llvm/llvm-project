; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s

; TODO: Update when spirv-val accepts casts from CodeSectionINTEL to Generic
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

define void @addrspacecast(ptr addrspace(9) %a) {
; CHECK: %[[#Int8:]] = OpTypeInt 8 0
; CHECK: %[[#Int8FnPtr:]] = OpTypePointer CodeSectionINTEL %[[#Int8]]
; CHECK: %[[#Int8GenericPtr:]] = OpTypePointer Generic %[[#Int8]]
; CHECK: %[[#FnParam:]] = OpFunctionParameter %[[#Int8FnPtr]]
; CHECK: OpPtrCastToGeneric %[[#Int8GenericPtr]] %[[#FnParam]]

  %res1 = addrspacecast ptr addrspace(9) %a to ptr addrspace(4)
  store i8 0, ptr addrspace(4) %res1
  ret void
}
