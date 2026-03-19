; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Image types may be represented in two ways while translating to SPIR-V:
; - OpenCL form, for example, '%opencl.image2d_ro_t',
; - SPIR-V form, for example, '%spirv.Image._void_1_0_0_0_0_0_0',
; but it is still one type which should be translated to one SPIR-V type.
;
; The test checks that the code below is successfully translated and only one
; SPIR-V type for images is generated (no duplicate OpTypeImage instructions).

; CHECK:     %[[#]] = OpTypeImage %[[#]] 2D
; CHECK-NOT: %[[#]] = OpTypeImage %[[#]] 2D

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(ptr addrspace(1), ptr addrspace(2), <2 x float>, float)

define spir_kernel void @read_image(ptr addrspace(1) %srcimg, ptr addrspace(2) %sampler){
entry:
  %spirvimg.addr = alloca target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), align 8
  %val = call <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_ff(ptr addrspace(1) %srcimg, ptr addrspace(2) %sampler, <2 x float> zeroinitializer, float 0.0)
  ret void
}
