; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A global variable keeps its concrete array pointee type even when its
; initializer is poison and it is non-constant, so a dynamic index is lowered as
; an OpAccessChain rather than dropped (which would make every invocation access
; element 0).

@tile = internal addrspace(3) global [64 x i32] poison, align 16

; CHECK-DAG: %[[#U32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Arr:]] = OpTypeArray %[[#U32]] %[[#]]
; CHECK-DAG: %[[#ArrPtr:]] = OpTypePointer Workgroup %[[#Arr]]
; CHECK-DAG: %[[#EltPtr:]] = OpTypePointer Workgroup %[[#U32]]
; CHECK-DAG: %[[#Tile:]] = OpVariable %[[#ArrPtr]] Workgroup

; The dynamic index survives as an OpAccessChain into the array tile and the
; variable is not collapsed to a scalar pointer.
; CHECK: %[[#Ptr:]] = OpAccessChain %[[#EltPtr]] %[[#Tile]] %[[#]]
; CHECK: OpStore %[[#Ptr]] %[[#]]

define void @store_dynamic_index(i32 %id) {
entry:
  %p = getelementptr i32, ptr addrspace(3) @tile, i32 %id
  store i32 42, ptr addrspace(3) %p, align 4
  ret void
}
