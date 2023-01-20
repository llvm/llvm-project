; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability Sampled1D
; CHECK-SPIRV-DAG: OpCapability SampledBuffer

%opencl.image1d_t = type opaque
%opencl.image1d_buffer_t = type opaque

define spir_kernel void @image_d(%opencl.image1d_t addrspace(1)* %image1d_td6, %opencl.image1d_buffer_t addrspace(1)* %image1d_buffer_td8) {
entry:
  ret void
}
