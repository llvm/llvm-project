; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

%opencl.image2d_t = type opaque

; CHECK: OpDecorate %[[#ID:]] LinkageAttributes "imageSampler" Export
; CHECK: %[[#ID]] = OpVariable %[[#]] UniformConstant %[[#]]

@imageSampler = addrspace(2) constant i32 36, align 4

define spir_kernel void @sample_kernel(%opencl.image2d_t addrspace(1)* %input, float addrspace(1)* nocapture %xOffsets, float addrspace(1)* nocapture %yOffsets, <4 x float> addrspace(1)* nocapture %results) {
  %1 = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %2 = trunc i64 %1 to i32
  %3 = tail call spir_func i64 @_Z13get_global_idj(i32 1)
  %4 = trunc i64 %3 to i32
  %5 = tail call spir_func i32 @_Z15get_image_width11ocl_image2d(%opencl.image2d_t addrspace(1)* %input)
  %6 = mul nsw i32 %4, %5
  %7 = add nsw i32 %6, %2
  %8 = sitofp i32 %2 to float
  %9 = insertelement <2 x float> undef, float %8, i32 0
  %10 = sitofp i32 %4 to float
  %11 = insertelement <2 x float> %9, float %10, i32 1
  %12 = tail call spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_f(%opencl.image2d_t addrspace(1)* %input, i32 36, <2 x float> %11)
  %13 = sext i32 %7 to i64
  %14 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %results, i64 %13
  store <4 x float> %12, <4 x float> addrspace(1)* %14, align 16
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func i32 @_Z15get_image_width11ocl_image2d(%opencl.image2d_t addrspace(1)*)

declare spir_func <4 x float> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_f(%opencl.image2d_t addrspace(1)*, i32, <2 x float>)
