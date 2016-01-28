; RUN: llc -march=amdgcn -mcpu=kaveri -mtriple=amdgcn-unknown-amdhsa -mattr=-fp32-denormals,+fp64-denormals < %s | FileCheck -check-prefix=FP64-DENORMAL -check-prefix=COMMON %s

; COMMON-LABEL: {{^}}test_kernel:
; COMMON-DENORMAL: compute_pgm_rsrc1_float_mode = compute_pgm_rsrc1_float_mode = 192
; COMMON-DENORMAL: compute_pgm_rsrc1_dx10_clamp = 1
define void @test_kernel(float addrspace(1)* %out0, double addrspace(1)* %out1) nounwind {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}
