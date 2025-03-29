; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-library %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate [[IntBufferVar:%[0-9]+]] DescriptorSet 16
; CHECK-DAG: OpDecorate [[IntBufferVar]] Binding 7
; CHECK-DAG: OpDecorate [[FloatBufferVar:%[0-9]+]] DescriptorSet 16
; CHECK-DAG: OpDecorate [[FloatBufferVar]] Binding 7

; CHECK-DAG: [[float:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[int:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[RWBufferTypeInt:%[0-9]+]] = OpTypeImage [[int]] Buffer 2 0 0 2 R32i {{$}}
; CHECK-DAG: [[RWBufferTypeFloat:%[0-9]+]] = OpTypeImage [[float]] Buffer 2 0 0 2 R32f {{$}}
; CHECK-DAG: [[IntBufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[RWBufferTypeInt]]
; CHECK-DAG: [[FloatBufferPtrType:%[0-9]+]] = OpTypePointer UniformConstant [[RWBufferTypeFloat]]
; CHECK-DAG: [[IntBufferVar]] = OpVariable [[IntBufferPtrType]] UniformConstant
; CHECK-DAG: [[FloatBufferVar]] = OpVariable [[FloatBufferPtrType]] UniformConstant

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @RWBufferLoad() #0 {
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer0 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer0, i32 0)
  store i32 0, ptr %ptr0, align 4

; Make sure we use the same variable with multiple loads.
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer1 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer1, i32 0)
  store i32 0, ptr %ptr1, align 4
  ret void
}

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @UseDifferentGlobalVar() #0 {
; Make sure we use a different variable from the first function. They have
; different types.
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeFloat]] [[FloatBufferVar]]
  %buffer0 = call target("spirv.Image", float, 5, 2, 0, 0, 2, 3)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_3(
          i32 16, i32 7, i32 1, i32 0, i1 false)
  %ptr0 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", float, 5, 2, 0, 0, 2, 3) %buffer0, i32 0)
  store float 0.0, ptr %ptr0, align 4
  ret void
}

; CHECK: {{%[0-9]+}} = OpFunction {{%[0-9]+}} DontInline {{%[0-9]+}}
; CHECK-NEXT: OpLabel
define void @ReuseGlobalVarFromFirstFunction() #0 {
; Make sure we use the same variable as the first function. They should be the
; same in case one function calls the other.
; CHECK: [[buffer:%[0-9]+]] = OpLoad [[RWBufferTypeInt]] [[IntBufferVar]]
  %buffer1 = call target("spirv.Image", i32, 5, 2, 0, 0, 2, 24)
      @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_24(
          i32 16, i32 7, i32 1, i32 0, i1 false)
  %ptr1 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.spv.resource.getpointer.p0.tspirv.Image_f32_5_2_0_0_2_0t(target("spirv.Image", i32, 5, 2, 0, 0, 2, 24) %buffer1, i32 0)
  store i32 0, ptr %ptr1, align 4
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
