; RUN: llc -O3 -verify-machineinstrs -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability StorageImageReadWithoutFormat

; CHECK-DAG: OpDecorate [[IntBufferVar:%[0-9]+]] DescriptorSet 16
; CHECK-DAG: OpDecorate [[IntBufferVar]] Binding 7

; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[zero:%[0-9]+]] = OpConstant [[int]] 0
; CHECK-DAG: [[v4_int:%[0-9]+]] = OpTypeVector [[int]] 4
; CHECK-DAG: [[RWBufferTypeInt:%[0-9]+]] = OpTypeImage [[int]] Buffer 2 0 0 2 R32i {{$}}
; CHECK-DAG: [[IntBufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[RWBufferTypeInt]]
; CHECK-DAG: [[IntBufferVar]] = OpVariable [[IntBufferPtrType]] UniformConstant


; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
declare <4 x i32> @get_data() #1

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @RWBufferStore_Vec4_I32() #0 {
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer0 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_i32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false)

; CHECK: [[data:%[0-9]+]] = OpFunctionCall
  %data = call <4 x i32> @get_data()
; CHECK: OpImageWrite [[buffer]] [[zero]] [[data]]
  call void @llvm.spv.resource.store.typedbuffer(target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer0, i32 0, <4 x i32> %data)

  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent noinline norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }