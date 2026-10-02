; The cooperative-matrix operations (load, store, mul-add, splat-construct) under
; the Vulkan/Shader flavor, reached via llvm.spv.cooperative.matrix.* intrinsics
; and GlobalISel selection because the OpenCL __spirv_CooperativeMatrix* builtin
; path is OpenCL-only.
;
; A single matmul tile fragment: C = A * B + 0. A is MatrixA (use 0), B is MatrixB
; (use 1), C the accumulator (use 2); all 16x16 f32, Subgroup scope (3). Memory
; layout 0 = RowMajorKHR, stride 16.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_vulkan_memory_model %s -o - | FileCheck %s

; CHECK-DAG: OpCapability CooperativeMatrixKHR
; CHECK-DAG: OpExtension "SPV_KHR_cooperative_matrix"
; The module sets the Vulkan memory model explicitly via spirv.MemoryModel
; metadata (spirv-val rejects Shader + CooperativeMatrixKHR under GLSL450).
; CHECK-DAG: OpCapability VulkanMemoryModelKHR
; CHECK-DAG: OpExtension "SPV_KHR_vulkan_memory_model"
; CHECK-DAG: OpMemoryModel Logical VulkanKHR
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; The three matrix types (A use 0, B use 1, C accumulator use 2) all lower to
; OpTypeCooperativeMatrixKHR over the f32 component type.
; CHECK-DAG: OpTypeCooperativeMatrixKHR %[[#F32]]
; Data flow: mul-add consumes both loaded operands plus the splat-constructed
; accumulator; the store consumes the mul-add result.
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

; AddressingModel=Logical (0), MemoryModel=VulkanKHR (3)
!spirv.MemoryModel = !{!0}
!0 = !{i32 0, i32 3}
