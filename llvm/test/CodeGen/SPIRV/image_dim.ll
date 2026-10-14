; Test OpTypeImage dimension-based capability requirements.
; target("spirv.Image", SampledType, Dim, Depth, Arrayed, MS, Sampled, Format)

; RUN: split-file %s %t

;; OpenCL: 1D and Buffer sampled images require Sampled1D and SampledBuffer.
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %t/opencl.ll -o - | FileCheck %s --check-prefix=CHECK-OPENCL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %t/opencl.ll -o - -filetype=obj | spirv-val %}

;; Vulkan: 2D multisampled storage images require StorageImageMultisample;
;;         2D multisampled arrayed images additionally require ImageMSArray.
;;         3D images require no extra capabilities.
; RUN: llc -O0 -mtriple=spirv-vulkan-library %t/vulkan.ll -o - | FileCheck %s --check-prefix=CHECK-VULKAN
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %t/vulkan.ll -o - -filetype=obj | spirv-val %}

; CHECK-OPENCL-DAG: OpCapability Sampled1D
; CHECK-OPENCL-DAG: OpCapability SampledBuffer

; CHECK-VULKAN-DAG: OpCapability StorageImageMultisample
; CHECK-VULKAN-DAG: OpCapability ImageMSArray
; CHECK-VULKAN-NOT: OpCapability ImageCubeArray

;--- opencl.ll
define spir_kernel void @test_image_dim(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %image1d, target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 0) %image1d_buffer) {
  ret void
}

;--- vulkan.ll
define void @test_2d_ms_storage(
  target("spirv.Image", float, 1, 0, 0, 1, 2, 3) %image2d_ms_storage
) #0 {
  ret void
}

define void @test_2d_ms_arrayed_storage(
  target("spirv.Image", float, 1, 0, 1, 1, 2, 3) %image2d_ms_arrayed_storage
) #0 {
  ret void
}

define void @test_3d(
  target("spirv.Image", float, 2, 0, 0, 0, 2, 3) %image3d_storage
) #0 {
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" }
