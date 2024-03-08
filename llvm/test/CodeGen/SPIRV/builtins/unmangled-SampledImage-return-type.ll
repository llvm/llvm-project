; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; XFAIL: *

; CHECK: %[[#image1d_t:]] = OpTypeImage
; CHECK: %[[#sampler_t:]] = OpTypeSampler
; CHECK: %[[#sampled_image_t:]] = OpTypeSampledImage

declare dso_local spir_func ptr addrspace(4) @__spirv_SampledImage(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %0, target("spirv.Sampler") %1) local_unnamed_addr

declare dso_local spir_func <4 x float> @__spirv_ImageSampleExplicitLod(ptr addrspace(4) %0, i32 %1, i32 %2, float %loaded) local_unnamed_addr

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(2) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @_ZTS17image_kernel_readILi1EE(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler")) {
; CHECK: OpFunction
; CHECK: %[[#image:]] = OpFunctionParameter %[[#image1d_t]]
; CHECK: %[[#sampler:]] = OpFunctionParameter %[[#sampler_t]]
  %loaded = load <3 x i64>, ptr addrspace(2) @__spirv_BuiltInGlobalInvocationId, align 32
  %extracted = extractelement <3 x i64> %loaded, i64 0
  %truncated = trunc i64 %extracted to i32
  %call1 = call spir_func ptr addrspace(4) @__spirv_SampledImage(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %0, target("spirv.Sampler") %1)
  %call2 = call spir_func <4 x float> @__spirv_ImageSampleExplicitLod(ptr addrspace(4) %call1, i32 %truncated, i32 2, float 0.000000e+00)

; CHECK: %[[#sampled_image:]] = OpSampledImage %[[#sampled_image_t]] %[[#image]] %[[#sampler]]
; CHECK: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#sampled_image]] %[[#]] {{.*}} %[[#]]

  ret void
}
