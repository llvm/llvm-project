;; __kernel void sample_test(read_only image2d_t src, read_only image1d_buffer_t buff) {}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: OpCapability Shader

%opencl.image2d_ro_t = type opaque
%opencl.image1d_buffer_ro_t = type opaque

define spir_kernel void @sample_test(%opencl.image2d_ro_t addrspace(1)* %src, %opencl.image1d_buffer_ro_t addrspace(1)* %buf) {
entry:
  ret void
}
