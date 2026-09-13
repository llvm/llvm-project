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

;; Vulkan: a cube image requires no extra capabilities unless it is arrayed;
;;         an arrayed cube image requires SampledCubeArray when it is sampled,
;;         and ImageCubeArray when it is a storage image.
; RUN: llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-cube.ll -o - | FileCheck %s --check-prefix=CHECK-CUBE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-cube.ll -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-cube-sampled.ll -o - | FileCheck %s --check-prefix=CHECK-CUBE-SAMPLED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-cube-sampled.ll -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-cube-storage.ll -o - | FileCheck %s --check-prefix=CHECK-CUBE-STORAGE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %t/vulkan-cube-storage.ll -o - -filetype=obj | spirv-val %}

; CHECK-OPENCL-DAG: OpCapability Sampled1D
; CHECK-OPENCL-DAG: OpCapability SampledBuffer

; CHECK-VULKAN-DAG: OpCapability StorageImageMultisample
; CHECK-VULKAN-DAG: OpCapability ImageMSArray
; CHECK-VULKAN-NOT: OpCapability ImageCubeArray

;; A cube image that is not arrayed needs neither cube-array capability,
;; whether it is sampled or a storage image.
; CHECK-CUBE-DAG: OpCapability Shader
; CHECK-CUBE-NOT: OpCapability SampledCubeArray
; CHECK-CUBE-NOT: OpCapability ImageCubeArray

; CHECK-CUBE-SAMPLED-DAG: OpCapability Shader
; CHECK-CUBE-SAMPLED-DAG: OpCapability SampledCubeArray
; CHECK-CUBE-SAMPLED-NOT: OpCapability ImageCubeArray

; CHECK-CUBE-STORAGE-DAG: OpCapability Shader
; CHECK-CUBE-STORAGE-DAG: OpCapability ImageCubeArray
; CHECK-CUBE-STORAGE-NOT: OpCapability SampledCubeArray

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

;--- vulkan-cube.ll
define void @test_cube_sampled(
  target("spirv.Image", float, 3, 0, 0, 0, 1, 0) %imagecube_sampled
) #0 {
  ret void
}

define void @test_cube_storage(
  target("spirv.Image", float, 3, 0, 0, 0, 2, 3) %imagecube_storage
) #0 {
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" }

;--- vulkan-cube-sampled.ll
define void @test_cube_arrayed_sampled(
  target("spirv.Image", float, 3, 0, 1, 0, 1, 0) %imagecube_arrayed_sampled
) #0 {
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" }

;--- vulkan-cube-storage.ll
define void @test_cube_arrayed_storage(
  target("spirv.Image", float, 3, 0, 1, 0, 2, 3) %imagecube_arrayed_storage
) #0 {
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" }
