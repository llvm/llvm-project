;; OpenCL C source
;; -----------------------------------------------
;; double d = 1.0;
;; kernel void test(read_only image2d_t img) {}
;; -----------------------------------------------

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

%opencl.image2d_t = type opaque

@d = addrspace(1) global double 1.000000e+00, align 8

define spir_kernel void @test(%opencl.image2d_t addrspace(1)* nocapture %img) {
entry:
  ret void
}

; CHECK-SPIRV-DAG: OpCapability Float64
; CHECK-SPIRV-DAG: OpCapability ImageBasic
