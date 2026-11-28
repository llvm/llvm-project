; A null element of a constant array or vector whose elements point to i32 (@g)
; is a null of that element type, as all elements of a composite share it.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#IntPtr:]] = OpTypePointer CrossWorkgroup %[[#Int]]
; CHECK-DAG: %[[#Arr:]] = OpTypeArray %[[#IntPtr]] %[[#]]
; CHECK-DAG: %[[#Vec:]] = OpTypeVector %[[#IntPtr]] 2
; CHECK-DAG: %[[#Null:]] = OpConstantNull %[[#IntPtr]]
; CHECK-DAG: OpSpecConstantComposite %[[#Arr]] %[[#]] %[[#Null]]
; CHECK-DAG: OpSpecConstantComposite %[[#Vec]] %[[#]] %[[#Null]]

@g = addrspace(1) global i32 0
@arr = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) @g, ptr addrspace(1) null]

define spir_kernel void @extract(ptr addrspace(1) %out) {
  %p = extractvalue [2 x ptr addrspace(1)] [ptr addrspace(1) @g, ptr addrspace(1) null], 1
  %v = load i32, ptr addrspace(1) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define spir_kernel void @extract_vector(ptr addrspace(1) %out) {
  %p = extractelement <2 x ptr addrspace(1)> <ptr addrspace(1) @g, ptr addrspace(1) null>, i32 1
  %v = load i32, ptr addrspace(1) %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}
