; RUN: llc -verify-machineinstrs -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; CHECK: OpName %[[#LDS:]] "lds"
; CHECK: OpDecorate %[[#LDS]] LinkageAttributes "lds" Import
; CHECK: %[[#UINT:]] = OpTypeInt 32 0
; CHECK: %[[#UINT_MAX:]] = OpConstant %[[#UINT]] 4294967295
; CHECK: %[[#LDS_ARR_TY:]] = OpTypeArray %[[#UINT]] %[[#UINT_MAX]]
; CHECK: %[[#LDS_ARR_PTR_WG:]] = OpTypePointer Workgroup %[[#LDS_ARR_TY]]
; CHECK: %[[#LDS]] = OpVariable %[[#LDS_ARR_PTR_WG]] Workgroup

@lds = external addrspace(3) global [0 x i32]

define spir_kernel void @foo(ptr addrspace(4) %in, ptr addrspace(4) %out) {
entry:
  %val = load i32, ptr addrspace(4) %in
  %add = add i32 %val, 1
  store i32 %add, ptr addrspace(4) %out
  ret void
}
