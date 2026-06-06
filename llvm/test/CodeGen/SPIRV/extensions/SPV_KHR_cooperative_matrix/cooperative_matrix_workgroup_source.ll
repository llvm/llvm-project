; A cooperative-matrix load/store whose source is a workgroup-shared array tile.
; In opaque-pointer IR the source pointer is the whole Workgroup array, so the
; selector must access-chain the array to element 0 before the cooperative-matrix
; op, which requires a pointer to the element type.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - -filetype=obj | spirv-val %}

@tile = internal addrspace(3) global [256 x half] poison, align 16

; CHECK-DAG: %[[#Half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#U32:]] = OpTypeInt 32 0
; The poison, non-constant Workgroup global keeps its array type (it is not
; collapsed to a scalar pointer).
; CHECK-DAG: %[[#Arr:]] = OpTypeArray %[[#Half]] %[[#]]
; CHECK-DAG: %[[#ArrPtr:]] = OpTypePointer Workgroup %[[#Arr]]
; CHECK-DAG: %[[#EltPtr:]] = OpTypePointer Workgroup %[[#Half]]
; CHECK-DAG: %[[#Tile:]] = OpVariable %[[#ArrPtr]] Workgroup
; CHECK-DAG: %[[#Zero:]] = OpConstant %[[#U32]] 0

; The array tile is access-chained to element 0 before the load, and that scalar
; pointer is what the cooperative-matrix load consumes.
; CHECK: %[[#LdPtr:]] = OpAccessChain %[[#EltPtr]] %[[#Tile]] %[[#Zero]]
; CHECK: %[[#Mat:]] = OpCooperativeMatrixLoadKHR %[[#]] %[[#LdPtr]]
; CHECK: %[[#StPtr:]] = OpAccessChain %[[#EltPtr]] %[[#Tile]] %[[#Zero]]
; CHECK: OpCooperativeMatrixStoreKHR %[[#StPtr]] %[[#Mat]]

define void @coop_matrix_workgroup_source() #0 {
entry:
  %m = call target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0)
       @llvm.spv.cooperative.matrix.load(ptr addrspace(3) @tile, i32 0, i32 16)
  call void
       @llvm.spv.cooperative.matrix.store(ptr addrspace(3) @tile,
       target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) %m, i32 0, i32 16)
  ret void
}

attributes #0 = { "hlsl.shader"="compute" "hlsl.numthreads"="1,1,1" }
