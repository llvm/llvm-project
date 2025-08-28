; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

@.str.b0 = private unnamed_addr constant [3 x i8] c"B0\00", align 1

; CHECK-NOT: OpCapability StorageImageReadWithoutFormat

; CHECK-DAG: OpDecorate [[IntBufferVar:%[0-9]+]] DescriptorSet 16
; CHECK-DAG: OpDecorate [[IntBufferVar]] Binding 7

; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[int]] 0
; CHECK-DAG: [[v4_int:%[0-9]+]] = OpTypeVector [[int]] 4
; CHECK-DAG: [[v2_int:%[0-9]+]] = OpTypeVector [[int]] 2
; CHECK-DAG: [[RWBufferTypeInt:%[0-9]+]] = OpTypeImage [[int]] Buffer 2 0 0 2 R32i {{$}}
; CHECK-DAG: [[IntBufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[RWBufferTypeInt]]
; CHECK-DAG: [[IntBufferVar]] = OpVariable [[IntBufferPtrType]] UniformConstant

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @RWBufferLoad_Vec4_I32() #0 {
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer0 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false, ptr nonnull @.str.b0)

; CHECK: OpImageRead [[v4_int]] [[buffer]] [[zero]]
  %data0 = call <4 x i32> @llvm.spv.resource.load.typedbuffer(
      target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer0, i32 0)

  ret void
}

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @RWBufferLoad_I32() #0 {
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer1 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false, ptr nonnull @.str.b0)

; CHECK: [[V:%[0-9]+]] = OpImageRead [[v4_int]] [[buffer]] [[zero]]
; CHECK: OpCompositeExtract [[int]] [[V]] 0
  %data1 = call i32 @llvm.spv.resource.load.typedbuffer(
      target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer1, i32 0)

  ret void
}

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @RWBufferLoad_Vec2_I32() #0 {
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer0 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false, ptr nonnull @.str.b0)

; CHECK: [[V:%[0-9]+]] = OpImageRead [[v4_int]] [[buffer]] [[zero]]
; CHECK: [[e0:%[0-9]+]] = OpCompositeExtract [[int]] [[V]] 0
; CHECK: [[e1:%[0-9]+]] = OpCompositeExtract [[int]] [[V]] 1
; CHECK: OpCompositeConstruct [[v2_int]] [[e0]] [[e1]]
  %data0 = call <2 x i32> @llvm.spv.resource.load.typedbuffer(
      target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer0, i32 0)

  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
