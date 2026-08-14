; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability StorageImageExtendedFormats
; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#image:]] = OpTypeImage %[[#uint]] Buffer 2 0 0 2 Rg32ui
; CHECK-DAG: %[[#ptr:]] = OpTypePointer UniformConstant %[[#image]]
; CHECK-DAG: %[[#var:]] = OpVariable %[[#ptr]] UniformConstant

@.str = private unnamed_addr constant [2 x i8] c"B\00", align 1

define void @main() #0 {
  %buffer = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 35)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_35(
          i32 16, i32 7, i32 1, i32 0, ptr nonnull @.str)
  %data = call <4 x i32> @llvm.spv.resource.load.typedbuffer(
      target("spirv.Image", i32, 5, 2, 0, 0, 2, 35) %buffer, i32 0)
  ret void
}

declare <4 x i32> @llvm.spv.resource.load.typedbuffer(target("spirv.Image", i32, 5, 2, 0, 0, 2, 35), i32)

attributes #0 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
