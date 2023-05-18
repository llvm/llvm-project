; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability Sampled1D
; CHECK-SPIRV-DAG: OpCapability SampledBuffer

define spir_kernel void @test_image_dim(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %image1d, target("spirv.Image", void, 5, 0, 0, 0, 0, 0, 0) %image1d_buffer) {
  ret void
}
