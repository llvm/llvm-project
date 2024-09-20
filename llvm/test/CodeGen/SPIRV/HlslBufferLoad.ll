; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate [[BufferVar:%[0-9]+]] DescriptorSet 16
; CHECK-DAG: OpDecorate [[BufferVar]] Binding 7

; CHECK: [[float:%[0-9]+]] = OpTypeFloat 32
; CHECK: [[RWBufferType:%[0-9]+]] = OpTypeImage [[float]] Buffer 2 0 0 2 R32i {{$}}
; CHECK: [[BufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[RWBufferType]]
; CHECK: [[BufferVar]] = OpVariable [[BufferPtrType]] UniformConstant

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @RWBufferLoad() #0 {
; CHECK-NEXT: [[buffer:%[0-9]+]] = OpLoad [[RWBufferType]] [[BufferVar]]
  %buffer0 = call target("spirv.Image", float, 5, 2, 0, 0, 2, 24)
      @llvm.spv.handle.fromBinding.tspirv.Image_f32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false)

; Make sure we use the same variable with multiple loads.
; CHECK-NEXT: [[buffer:%[0-9]+]] = OpLoad [[RWBufferType]] [[BufferVar]]
  %buffer1 = call target("spirv.Image", float, 5, 2, 0, 0, 2, 24)
      @llvm.spv.handle.fromBinding.tspirv.Image_f32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false)
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }