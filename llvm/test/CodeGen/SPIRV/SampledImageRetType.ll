; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

%opencl.image1d_ro_t = type opaque
; CHECK: %[[#image1d_t:]] = OpTypeImage
%opencl.sampler_t = type opaque
; CHECK: %[[#sampler_t:]] = OpTypeSampler
; CHECK: %[[#sampled_image_t:]] = OpTypeSampledImage

declare dso_local spir_func i8 addrspace(4)* @_Z20__spirv_SampledImageI14ocl_image1d_roPvET0_T_11ocl_sampler(%opencl.image1d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*) local_unnamed_addr

declare dso_local spir_func <4 x float> @_Z30__spirv_ImageSampleExplicitLodIPvDv4_fiET0_T_T1_if(i8 addrspace(4)*, i32, i32, float) local_unnamed_addr

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(2) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @_ZTS17image_kernel_readILi1EE(%opencl.image1d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*) {
; CHECK: OpFunction
; CHECK: %[[#image:]] = OpFunctionParameter %[[#image1d_t]]
; CHECK: %[[#sampler:]] = OpFunctionParameter %[[#sampler_t]]
  %3 = load <3 x i64>, <3 x i64> addrspace(2)* @__spirv_BuiltInGlobalInvocationId, align 32
  %4 = extractelement <3 x i64> %3, i64 0
  %5 = trunc i64 %4 to i32
  %6 = tail call spir_func i8 addrspace(4)* @_Z20__spirv_SampledImageI14ocl_image1d_roPvET0_T_11ocl_sampler(%opencl.image1d_ro_t addrspace(1)* %0, %opencl.sampler_t addrspace(2)* %1)
  %7 = tail call spir_func <4 x float> @_Z30__spirv_ImageSampleExplicitLodIPvDv4_fiET0_T_T1_if(i8 addrspace(4)* %6, i32 %5, i32 2, float 0.000000e+00)

; CHECK: %[[#sampled_image:]] = OpSampledImage %[[#sampled_image_t]] %[[#image]] %[[#sampler]]
; CHECK: %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#sampled_image]] %[[#]] {{.*}} %[[#]]

  ret void
}
