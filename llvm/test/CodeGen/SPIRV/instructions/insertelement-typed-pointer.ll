; @g (i32 pointee) inserted into a pointer vector is bitcast to
; the vector's element type (pointer to i8).

; RUN: llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_INTEL_masked_gather_scatter -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#P8:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; CHECK-DAG: %[[#V:]] = OpTypeVector %[[#P8]] 2
; CHECK-DAG: %[[#G:]] = OpVariable %[[#]] CrossWorkgroup

; CHECK: OpFunction %[[#V]]
; CHECK: %[[#C:]] = OpBitcast %[[#P8]] %[[#G]]
; CHECK: OpCompositeInsert %[[#V]] %[[#C]] %[[#]] 0

@g = addrspace(1) global i32 0

define <2 x ptr addrspace(1)> @insert(ptr addrspace(1) %p) {
  %v0 = insertelement <2 x ptr addrspace(1)> poison, ptr addrspace(1) @g, i32 0
  %v1 = insertelement <2 x ptr addrspace(1)> %v0, ptr addrspace(1) %p, i32 1
  ret <2 x ptr addrspace(1)> %v1
}
