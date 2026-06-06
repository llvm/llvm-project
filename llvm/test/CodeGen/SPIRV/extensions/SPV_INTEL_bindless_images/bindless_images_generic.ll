; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bindless_images,+SPV_EXT_shader_image_int64 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bindless_images,+SPV_EXT_shader_image_int64 %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpConvertHandleTo[Image/Sampler/SampledImage]INTEL instruction
; CHECK-ERROR-SAME: require the following SPIR-V extension: SPV_INTEL_bindless_images

; CHECK-DAG: OpCapability BindlessImagesINTEL
; CHECK-DAG: OpCapability Int64ImageEXT
; CHECK-DAG: OpExtension "SPV_INTEL_bindless_images"
; CHECK-DAG: OpExtension "SPV_EXT_shader_image_int64"

; CHECK-DAG: %[[#VoidTy:]] = OpTypeVoid
; CHECK-DAG: %[[#Int64Ty:]] = OpTypeInt 64
; CHECK-DAG: %[[#Const42:]] = OpConstant %[[#Int64Ty]] 42
; CHECK-DAG: %[[#Const43:]] = OpConstant %[[#Int64Ty]] 43
; CHECK-DAG: %[[#IntImgTy:]] = OpTypeImage %[[#Int64Ty]]
; CHECK-DAG: %[[#SamplerTy:]] = OpTypeSampler
; CHECK-DAG: %[[#IntSmpImgTy:]] = OpTypeImage %[[#Int64Ty]]
; CHECK-DAG: %[[#SampImageTy:]] = OpTypeSampledImage %[[#IntSmpImgTy]]
; CHECK: %[[#Input:]] = OpFunctionParameter %[[#Int64Ty]]
; CHECK: %[[#]] = OpConvertHandleToImageINTEL %[[#IntImgTy]] %[[#Input]]
; CHECK: %[[#]] = OpConvertHandleToSamplerINTEL %[[#SamplerTy]] %[[#Const42]]
; CHECK: %[[#]] = OpConvertHandleToSampledImageINTEL %[[#SampImageTy]] %[[#Const43]]

define spir_func void @foo(i64 %in) {
  %img = call spir_func target("spirv.Image", i64, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELl(i64 %in)
  %samp = call spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64 42)
  %sampImage = call spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64 43)
  ret void
}

declare spir_func target("spirv.Image", i64, 2, 0, 0, 0, 0, 0, 0) @_Z33__spirv_ConvertHandleToImageINTELl(i64)

declare spir_func target("spirv.Sampler") @_Z35__spirv_ConvertHandleToSamplerINTELl(i64)

declare spir_func target("spirv.SampledImage", i64, 1, 0, 0, 0, 0, 0, 0) @_Z40__spirv_ConvertHandleToSampledImageINTELl(i64)
