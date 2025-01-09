; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Shader
; CHECK-NEXT: OpCapability InputAttachmentArrayDynamicIndexing
; SCHECK-NEXT: OpCapability InputAttachment
; CHECK-NOT: OpCapability

; CHECK-DAG: OpDecorate [[Var:%[0-9]+]] DescriptorSet 3
; CHECK-DAG: OpDecorate [[Var]] Binding 4

; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[BufferType:%[0-9]+]] = OpTypeImage [[int]] SubpassData 2 0 0 2 Unknown {{$}}
; CHECK-DAG: [[BufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[BufferType]]
; CHECK-DAG: [[ArraySize:%[0-9]+]] = OpConstant [[int]] 3
; CHECK-DAG: [[One:%[0-9]+]] = OpConstant [[int]] 1
; CHECK-DAG: [[Zero:%[0-9]+]] = OpConstant [[int]] 0
; CHECK-DAG: [[BufferArrayType:%[0-9]+]] = OpTypeArray [[BufferType]] [[ArraySize]]
; CHECK-DAG: [[ArrayPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[BufferArrayType]]
; CHECK-DAG: [[Var]] = OpVariable [[ArrayPtrType]] UniformConstant

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @main() #0 {
; CHECK: [[ac:%[0-9]+]] = OpAccessChain [[BufferPtrType]] [[Var]] [[Zero]]
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[BufferType]] [[ac]]
  %buffer0 = call target("spirv.Image", i32, 6, 2, 0, 0, 2, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_6_2_0_0_2_0(
          i32 3, i32 4, i32 3, i32 0, i1 false)

; CHECK: [[ac:%[0-9]+]] = OpAccessChain [[BufferPtrType]] [[Var]] [[One]]
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[BufferType]] [[ac]]
  %buffer1 = call target("spirv.Image", i32, 6, 2, 0, 0, 2, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_6_2_0_0_2_0(
          i32 3, i32 4, i32 3, i32 1, i1 false)
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
