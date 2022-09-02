;; Test CL opaque types
;;
;; // cl-types.cl
;; // CL source code for generating LLVM IR.
;; // Command for compilation:
;; //  clang -cc1 -x cl -cl-std=CL2.0 -triple spir-unknown-unknown -emit-llvm cl-types.cl
;; void kernel foo(
;;  read_only pipe int a,
;;  write_only pipe int b,
;;  read_only image1d_t c1,
;;  read_only image2d_t d1,
;;  read_only image3d_t e1,
;;  read_only image2d_array_t f1,
;;  read_only image1d_buffer_t g1,
;;  write_only image1d_t c2,
;;  read_write image2d_t d3,
;;  sampler_t s
;; ) {
;; }

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability Sampled1D
; CHECK-SPIRV-DAG: OpCapability SampledBuffer
; CHECK-SPIRV-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#PIPE_RD:]] = OpTypePipe ReadOnly
; CHECK-SPIRV-DAG: %[[#PIPE_WR:]] = OpTypePipe WriteOnly
; CHECK-SPIRV-DAG: %[[#IMG1D_RD:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG2D_RD:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG3D_RD:]] = OpTypeImage %[[#VOID]] 3D 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG2DA_RD:]] = OpTypeImage %[[#VOID]] 2D 0 1 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG1DB_RD:]] = OpTypeImage %[[#VOID]] Buffer 0 0 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#IMG1D_WR:]] = OpTypeImage %[[#VOID]] 1D 0 0 0 0 Unknown WriteOnly
; CHECK-SPIRV-DAG: %[[#IMG2D_RW:]] = OpTypeImage %[[#VOID]] 2D 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV-DAG: %[[#SAMP:]] = OpTypeSampler
; CHECK-SPIRV-DAG: %[[#SAMPIMG:]] = OpTypeSampledImage %[[#IMG2D_RD]]

; CHECK-SPIRV:     %[[#SAMP_CONST:]] = OpConstantSampler %[[#SAMP]] None 0 Linear

%opencl.pipe_ro_t = type opaque
%opencl.pipe_wo_t = type opaque
%opencl.image3d_ro_t = type opaque
%opencl.image2d_array_ro_t = type opaque
%opencl.image1d_buffer_ro_t = type opaque
%opencl.image1d_ro_t = type opaque
%opencl.image1d_wo_t = type opaque
%opencl.image2d_rw_t = type opaque
%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque

; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#PIPE_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#PIPE_WR]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG1D_RD]]
; CHECK-SPIRV: %[[#IMG_ARG:]] = OpFunctionParameter %[[#IMG2D_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG3D_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG2DA_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG1DB_RD]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG1D_WR]]
; CHECK-SPIRV: %[[#]] = OpFunctionParameter %[[#IMG2D_RW]]
; CHECK-SPIRV: %[[#SAMP_ARG:]] = OpFunctionParameter %[[#SAMP]]

define spir_kernel void @foo(
  %opencl.pipe_ro_t addrspace(1)* nocapture %a,
  %opencl.pipe_wo_t addrspace(1)* nocapture %b,
  %opencl.image1d_ro_t addrspace(1)* nocapture %c1,
  %opencl.image2d_ro_t addrspace(1)* nocapture %d1,
  %opencl.image3d_ro_t addrspace(1)* nocapture %e1,
  %opencl.image2d_array_ro_t addrspace(1)* nocapture %f1,
  %opencl.image1d_buffer_ro_t addrspace(1)* nocapture %g1,
  %opencl.image1d_wo_t addrspace(1)* nocapture %c2,
  %opencl.image2d_rw_t addrspace(1)* nocapture %d3,
  %opencl.sampler_t addrspace(2)* %s) {
entry:
; CHECK-SPIRV: %[[#SAMPIMG_VAR1:]] = OpSampledImage %[[#SAMPIMG]] %[[#IMG_ARG]] %[[#SAMP_ARG]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SAMPIMG_VAR1]]
  %.tmp = call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv4_if(%opencl.image2d_ro_t addrspace(1)* %d1, %opencl.sampler_t addrspace(2)* %s, <4 x i32> zeroinitializer, float 1.000000e+00)

; CHECK-SPIRV: %[[#SAMPIMG_VAR2:]] = OpSampledImage %[[#SAMPIMG]] %[[#IMG_ARG]] %[[#SAMP_CONST]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SAMPIMG_VAR2]]
  %0 = call %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32 32)
  %.tmp2 = call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv4_if(%opencl.image2d_ro_t addrspace(1)* %d1, %opencl.sampler_t addrspace(2)* %0, <4 x i32> zeroinitializer, float 1.000000e+00)
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv4_if(%opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, <4 x i32>, float)

declare %opencl.sampler_t addrspace(2)* @__translate_sampler_initializer(i32)
