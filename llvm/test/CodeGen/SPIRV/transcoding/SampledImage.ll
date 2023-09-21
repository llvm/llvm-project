; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; constant sampler_t constSampl = CLK_FILTER_LINEAR;
;;
;; __kernel
;; void sample_kernel_float(image2d_t input, float2 coords, global float4 *results, sampler_t argSampl) {
;;   *results = read_imagef(input, constSampl, coords);
;;   *results = read_imagef(input, argSampl, coords);
;;   *results = read_imagef(input, CLK_FILTER_NEAREST|CLK_ADDRESS_REPEAT, coords);
;; }
;;
;; __kernel
;; void sample_kernel_int(image2d_t input, float2 coords, global int4 *results, sampler_t argSampl) {
;;   *results = read_imagei(input, constSampl, coords);
;;   *results = read_imagei(input, argSampl, coords);
;;   *results = read_imagei(input, CLK_FILTER_NEAREST|CLK_ADDRESS_REPEAT, coords);
;; }

; CHECK-SPIRV: OpCapability LiteralSampler
; CHECK-SPIRV: OpName %[[#sample_kernel_float:]] "sample_kernel_float"
; CHECK-SPIRV: OpName %[[#sample_kernel_int:]] "sample_kernel_int"

; CHECK-SPIRV:     %[[#TypeSampler:]] = OpTypeSampler
; CHECK-SPIRV-DAG: %[[#SampledImageTy:]] = OpTypeSampledImage
; CHECK-SPIRV-DAG: %[[#ConstSampler1:]] = OpConstantSampler %[[#TypeSampler]] None 0 Linear
; CHECK-SPIRV-DAG: %[[#ConstSampler2:]] = OpConstantSampler %[[#TypeSampler]] Repeat 0 Nearest
; CHECK-SPIRV-DAG: %[[#ConstSampler3:]] = OpConstantSampler %[[#TypeSampler]] None 0 Linear
; CHECK-SPIRV-DAG: %[[#ConstSampler4:]] = OpConstantSampler %[[#TypeSampler]] Repeat 0 Nearest

; CHECK-SPIRV: %[[#sample_kernel_float]] = OpFunction %{{.*}}
; CHECK-SPIRV: %[[#InputImage:]] = OpFunctionParameter %{{.*}}
; CHECK-SPIRV: %[[#argSampl:]] = OpFunctionParameter %[[#TypeSampler]]

; CHECK-SPIRV: %[[#SampledImage1:]] = OpSampledImage %[[#SampledImageTy]] %[[#InputImage]] %[[#ConstSampler1]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SampledImage1]]

; CHECK-SPIRV: %[[#SampledImage2:]] = OpSampledImage %[[#SampledImageTy]] %[[#InputImage]] %[[#argSampl]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SampledImage2]]

; CHECK-SPIRV: %[[#SampledImage3:]] = OpSampledImage %[[#SampledImageTy]] %[[#InputImage]] %[[#ConstSampler2]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SampledImage3]]

define dso_local spir_kernel void @sample_kernel_float(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, <2 x float> noundef %coords, <4 x float> addrspace(1)* nocapture noundef writeonly %results, target("spirv.Sampler") %argSampl) local_unnamed_addr {
entry:
  %0 = tail call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 32)
  %call = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, target("spirv.Sampler") %0, <2 x float> noundef %coords)
  store <4 x float> %call, <4 x float> addrspace(1)* %results, align 16
  %call1 = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, target("spirv.Sampler") %argSampl, <2 x float> noundef %coords)
  store <4 x float> %call1, <4 x float> addrspace(1)* %results, align 16
  %1 = tail call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 22)
  %call2 = tail call spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, target("spirv.Sampler") %1, <2 x float> noundef %coords)
  store <4 x float> %call2, <4 x float> addrspace(1)* %results, align 16
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x float> noundef) local_unnamed_addr

declare spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32) local_unnamed_addr

; CHECK-SPIRV: %[[#sample_kernel_int]] = OpFunction %{{.*}}
; CHECK-SPIRV: %[[#InputImage:]] = OpFunctionParameter %{{.*}}
; CHECK-SPIRV: %[[#argSampl:]] = OpFunctionParameter %[[#TypeSampler]]

; CHECK-SPIRV: %[[#SampledImage4:]] = OpSampledImage %[[#SampledImageTy]] %[[#InputImage]] %[[#ConstSampler3]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SampledImage4]]

; CHECK-SPIRV: %[[#SampledImage5:]] = OpSampledImage %[[#SampledImageTy]] %[[#InputImage]] %[[#argSampl]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SampledImage5]]

; CHECK-SPIRV: %[[#SampledImage6:]] = OpSampledImage %[[#SampledImageTy]] %[[#InputImage]] %[[#ConstSampler4]]
; CHECK-SPIRV: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#SampledImage6]]

define dso_local spir_kernel void @sample_kernel_int(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, <2 x float> noundef %coords, <4 x i32> addrspace(1)* nocapture noundef writeonly %results, target("spirv.Sampler") %argSampl) local_unnamed_addr {
entry:
  %0 = tail call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 32)
  %call = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, target("spirv.Sampler") %0, <2 x float> noundef %coords)
  store <4 x i32> %call, <4 x i32> addrspace(1)* %results, align 16
  %call1 = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, target("spirv.Sampler") %argSampl, <2 x float> noundef %coords)
  store <4 x i32> %call1, <4 x i32> addrspace(1)* %results, align 16
  %1 = tail call spir_func target("spirv.Sampler") @__translate_sampler_initializer(i32 22)
  %call2 = tail call spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %input, target("spirv.Sampler") %1, <2 x float> noundef %coords)
  store <4 x i32> %call2, <4 x i32> addrspace(1)* %results, align 16
  ret void
}

declare spir_func <4 x i32> @_Z11read_imagei14ocl_image2d_ro11ocl_samplerDv2_f(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"), <2 x float> noundef) local_unnamed_addr
