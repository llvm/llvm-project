; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpCapability ImageReadWrite
; CHECK-SPIRV-DAG: OpCapability LiteralSampler
; CHECK-SPIRV-DAG: %[[#TyVoid:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#TyImageID:]] = OpTypeImage %[[#TyVoid]] 1D 0 0 0 0 Unknown ReadWrite
; CHECK-SPIRV-DAG: %[[#TySampledImageID:]] = OpTypeSampledImage %[[#TyImageID]]

; CHECK-SPIRV-DAG: %[[#ResID:]] = OpSampledImage %[[#TySampledImageID]]
; CHECK-SPIRV:     %[[#]] = OpImageSampleExplicitLod %[[#]] %[[#ResID]]

define spir_func void @sampFun(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2) %image) {
entry:
  %image.addr = alloca target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2), align 4
  store target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2) %image, ptr %image.addr, align 4
  %0 = load target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2), ptr %image.addr, align 4
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2) %0, i32 8, i32 2)
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 2), i32, i32)
