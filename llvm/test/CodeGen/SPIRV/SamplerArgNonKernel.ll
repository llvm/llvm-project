; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;CHECK: OpEntryPoint Kernel %[[#KernelId:]]
;CHECK: %[[#image2d_t:]] = OpTypeImage
;CHECK: %[[#sampler_t:]] = OpTypeSampler
;CHECK: %[[#sampled_image_t:]] = OpTypeSampledImage

define spir_func float @test(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %Img, target("spirv.Sampler") %Smp) {
;CHECK-NOT: %[[#KernelId]] = OpFunction %[[#]]
;CHECK: OpFunction
;CHECK: %[[#image:]] = OpFunctionParameter %[[#image2d_t]]
;CHECK: %[[#sampler:]] = OpFunctionParameter %[[#sampler_t]]
entry:
  %call = call spir_func <4 x i32> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %Img, target("spirv.Sampler") %Smp, <2 x i32> zeroinitializer)
;CHECK: %[[#sampled_image:]] = OpSampledImage %[[#sampled_image_t]] %[[#image]] %[[#sampler]]
;CHECK: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#sampled_image]] %[[#]] Lod %[[#]]

  %0 = extractelement <4 x i32> %call, i32 0
  %conv = sitofp i32 %0 to float
  ret float %conv
}

declare spir_func <4 x i32> @_Z11read_imagef11ocl_image2d11ocl_samplerDv2_i(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0), i32, <2 x i32>)

define spir_kernel void @test2(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %Img, target("spirv.Sampler") %Smp, ptr addrspace(1) %result) {
;CHECK: %[[#KernelId]] = OpFunction  %[[#]]
entry:
  %call = call spir_func float @test(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %Img, target("spirv.Sampler") %Smp)
  %0 = load float, ptr addrspace(1) %result, align 4
  %add = fadd float %0, %call
  store float %add, ptr addrspace(1) %result, align 4
  ret void
}
