; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#v4i32:]] = OpTypeVector %[[#i32]] 4
; CHECK-DAG: %[[#ptrv4i32:]] = OpTypePointer CrossWorkgroup %[[#v4i32]]
; CHECK-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#typesampled:]] = OpTypeSampledImage
; CHECK-DAG: %[[#const0:]] = OpConstant %[[#float]] 0
; CHECK: OpFunction
; CHECK: OpFunctionParameter
; CHECK: %[[#arg1:]] = OpFunctionParameter
; CHECK: %[[#arg2:]] = OpFunctionParameter
; CHECK: %[[#addr:]] = OpInBoundsPtrAccessChain
; CHECK: %[[#img:]] = OpSampledImage %[[#typesampled:]] %[[#arg1]] %[[#arg2]]
; CHECK: %[[#sample:]] = OpImageSampleExplicitLod %[[#v4i32]] %[[#img]] %[[#const0]] Lod %[[#const0]]
; CHECK: %[[#casted:]] = OpBitcast %[[#ptrv4i32]] %[[#addr]]
; CHECK: OpStore %[[#casted]] %[[#sample]] Aligned 16

%"class.sycl::_V1::vec" = type { <4 x i32> }

define weak_odr dso_local spir_kernel void @foo(ptr addrspace(1) align 16 %_arg_acc, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %_arg_img, target("spirv.Sampler") %_arg_sampler) {
entry:
  %data = getelementptr inbounds %"class.sycl::_V1::vec", ptr addrspace(1) %_arg_acc, i64 0
  %img = tail call spir_func target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImage(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %_arg_img, target("spirv.Sampler") %_arg_sampler)
  %sample = tail call spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLod(target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) %img, float 0.000000e+00, i32 2, float 0.000000e+00)
  store <4 x i32> %sample, ptr addrspace(1) %data, align 16
  ret void
}

declare dso_local spir_func target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) @_Z20__spirv_SampledImage(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0), target("spirv.Sampler"))
declare dso_local spir_func <4 x i32> @_Z30__spirv_ImageSampleExplicitLod(target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0), float, i32, float)
