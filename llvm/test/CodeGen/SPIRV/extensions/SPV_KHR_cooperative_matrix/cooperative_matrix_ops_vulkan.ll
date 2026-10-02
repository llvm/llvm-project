; Cooperative-matrix load / mul-add / splat / store under the Vulkan/Shader
; flavor, via llvm.spv.cooperative.matrix.* intrinsics and GlobalISel selection.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - | FileCheck %s

; CHECK-DAG: OpCapability CooperativeMatrixKHR
; CHECK-DAG: OpExtension "SPV_KHR_cooperative_matrix"
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: OpTypeCooperativeMatrixKHR %[[#F32]]
; CHECK: %[[#A:]] = OpCooperativeMatrixLoadKHR
; CHECK: %[[#B:]] = OpCooperativeMatrixLoadKHR
; CHECK: %[[#C0:]] = OpCompositeConstruct
; CHECK: %[[#C:]] = OpCooperativeMatrixMulAddKHR %[[#]] %[[#A]] %[[#B]] %[[#C0]]
; CHECK: OpCooperativeMatrixStoreKHR %[[#]] %[[#C]]

define spir_func void @matmul_tile(ptr %pa, ptr %pb, ptr %pc) {
entry:
  %a = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0)
       @llvm.spv.cooperative.matrix.load(ptr %pa, i32 0, i32 16)
  %b = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 1)
       @llvm.spv.cooperative.matrix.load(ptr %pb, i32 0, i32 16)
  %c0 = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
        @llvm.spv.cooperative.matrix.splat(float 0.000000e+00)
  %c = call target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
       @llvm.spv.cooperative.matrix.muladd(
         target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 0) %a,
         target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 1) %b,
         target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c0)
  call void @llvm.spv.cooperative.matrix.store(
         ptr %pc,
         target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %c,
         i32 0, i32 16)
  ret void
}
