; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#image1d_t:]] = OpTypeImage
; CHECK: %[[#sampler_t:]] = OpTypeSampler
; CHECK: %[[#sampled_image_t:]] = OpTypeSampledImage

declare dso_local spir_func ptr addrspace(4) @_Z20__spirv_SampledImageI14ocl_image1d_roPvET0_T_11ocl_sampler(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %0, target("spirv.Sampler") %1) local_unnamed_addr

declare dso_local spir_func <4 x float> @_Z30__spirv_ImageSampleExplicitLodIPvDv4_fiET0_T_T1_if(ptr addrspace(4) %0, i32 %1, i32 %2, float %3) local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) %0, float %1, i32  %2, float %3) local_unnamed_addr

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(2) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @_ZTS17image_kernel_readILi1EE(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler")) {
; CHECK: OpFunction
; CHECK: %[[#image:]] = OpFunctionParameter %[[#image1d_t]]
; CHECK: %[[#sampler:]] = OpFunctionParameter %[[#sampler_t]]
  %3 = load <3 x i64>, ptr addrspace(2) @__spirv_BuiltInGlobalInvocationId, align 32
  %4 = extractelement <3 x i64> %3, i64 0
  %5 = trunc i64 %4 to i32
  %6 = call spir_func ptr addrspace(4) @_Z20__spirv_SampledImageI14ocl_image1d_roPvET0_T_11ocl_sampler(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %0, target("spirv.Sampler") %1)
  %7 = call spir_func <4 x float> @_Z30__spirv_ImageSampleExplicitLodIPvDv4_fiET0_T_T1_if(ptr addrspace(4) %6, i32 %5, i32 2, float 0.000000e+00)

; CHECK: %[[#sampled_image:]] = OpSampledImage %[[#sampled_image_t]] %[[#image]] %[[#sampler]]
; CHECK: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#sampled_image]] %[[#]] {{.*}} %[[#]]

  ret void
}

define weak_odr dso_local spir_kernel void @foo_lod(target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) %_arg) {
  %lod = call spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_jfET0_T_T1_if(target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) %_arg, float 0x3FE7FFEB00000000, i32 2, float 0.000000e+00)
; CHECK: %[[#sampled_image_lod:]] = OpFunctionParameter %[[#sampled_image_t]]
; CHECK: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#sampled_image_lod]] %[[#]] {{.*}} %[[#]]
  ret void
}
