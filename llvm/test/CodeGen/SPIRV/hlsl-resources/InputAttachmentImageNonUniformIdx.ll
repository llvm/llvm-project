; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Shader
; CHECK-NEXT: OpCapability ShaderNonUniformEXT
; CHECK-NEXT: OpCapability InputAttachmentArrayNonUniformIndexing
; SCHECK-NEXT: OpCapability InputAttachment
; CHECK-NOT: OpCapability

; CHECK-DAG: OpDecorate [[Var:%[0-9]+]] DescriptorSet 3
; CHECK-DAG: OpDecorate [[Var]] Binding 4
; CHECK: OpDecorate [[Zero:%[0-9]+]] NonUniform
; CHECK: OpDecorate [[ac0:%[0-9]+]] NonUniform
; CHECK: OpDecorate [[ld0:%[0-9]+]] NonUniform
; CHECK: OpDecorate [[One:%[0-9]+]] NonUniform
; CHECK: OpDecorate [[ac1:%[0-9]+]] NonUniform
; CHECK: OpDecorate [[ld1:%[0-9]+]] NonUniform

; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[BufferType:%[0-9]+]] = OpTypeImage [[int]] SubpassData 2 0 0 2 Unknown {{$}}
; CHECK-DAG: [[BufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[BufferType]]
; CHECK-DAG: [[ArraySize:%[0-9]+]] = OpConstant [[int]] 3
; CHECK-DAG: [[One]] = OpConstant [[int]] 1
; CHECK-DAG: [[Zero]] = OpConstant [[int]] 0
; CHECK-DAG: [[BufferArrayType:%[0-9]+]] = OpTypeArray [[BufferType]] [[ArraySize]]
; CHECK-DAG: [[ArrayPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[BufferArrayType]]
; CHECK-DAG: [[Var]] = OpVariable [[ArrayPtrType]] UniformConstant

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @main() #0 {
; CHECK: [[ac0]] = OpAccessChain [[BufferPtrType]] [[Var]] [[Zero]]
; CHECK: [[ld0]] = OpLoad [[BufferType]] [[ac0]]
  %buffer0 = call target("spirv.Image", i32, 6, 2, 0, 0, 2, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_6_2_0_0_2_0(
          i32 3, i32 4, i32 3, i32 0, i1 true)

; CHECK: [[ac1:%[0-9]+]] = OpAccessChain [[BufferPtrType]] [[Var]] [[One]]
; CHECK: [[ld1]] = OpLoad [[BufferType]] [[ac1]]
  %buffer1 = call target("spirv.Image", i32, 6, 2, 0, 0, 2, 0)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_6_2_0_0_2_0(
          i32 3, i32 4, i32 3, i32 1, i1 true)
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
