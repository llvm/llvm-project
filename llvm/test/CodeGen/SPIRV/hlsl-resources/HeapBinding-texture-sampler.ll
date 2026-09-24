; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val --target-env spv1.6 %}

; This test declares storage buffer Buf at descriptor set 0, binding 1, and
; creates one texture and two samplers by dynamically indexing into descriptor
; heaps. SPIR-V legalization creates separate runtime arrays for resource and
; sampler descriptors heaps, assigning the resource heap to binding 0 and the
; sampler heap to binding 2 (the first two available bindings).

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

; CHECK-DAG: OpCapability RuntimeDescriptorArrayEXT

; CHECK-DAG: OpName [[Buf:%[0-9]+]] "Buf"
; CHECK-DAG: OpDecorate [[Buf]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[Buf]] Binding 1

; CHECK-DAG: OpName [[ResourceHeap:%[0-9]+]] "ResourceDescriptorHeap"
; CHECK-DAG: OpDecorate [[ResourceHeap]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[ResourceHeap]] Binding 0

; CHECK-DAG: OpName [[SamplerHeap:%[0-9]+]] "SamplerDescriptorHeap"
; CHECK-DAG: OpDecorate [[SamplerHeap]] DescriptorSet 0
; CHECK-DAG: OpDecorate [[SamplerHeap]] Binding 2

; Types of descriptor heap arrays

; CHECK: [[Float32:%[0-9]+]] = OpTypeFloat 32
; CHECK: [[SamplerType:%[0-9]+]] = OpTypeSampler
; CHECK: [[ImageType:%[0-9]+]] = OpTypeImage [[Float32]] 2D 2 0 0 1 Unknown
; CHECK: [[ResourceRTArray:%[0-9]+]] = OpTypeRuntimeArray [[ImageType]]
; CHECK: [[ResourceRTArrayPtr:%[0-9]+]] = OpTypePointer UniformConstant [[ResourceRTArray]]
; CHECK: [[SamplerRTArray:%[0-9]+]] = OpTypeRuntimeArray [[SamplerType]]
; CHECK: [[SamplerRTArrayPtr:%[0-9]+]] = OpTypePointer UniformConstant [[SamplerRTArray]]

; CHECK: [[ResourceHeap]] = OpVariable [[ResourceRTArrayPtr]] UniformConstant
; CHECK: [[SamplerHeap]] = OpVariable [[SamplerRTArrayPtr]] UniformConstant

define void @test(i32 %TexIndex, i32 %Samp0Index, i32 %Samp1Index) {
entry:
  %Buf = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_1t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str)

  %Texture = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefromheap.tspirv.Image_f32_1_2_0_0_1_0t(i32 %TexIndex)
  %Samp0 = tail call target("spirv.Sampler") @llvm.spv.resource.handlefromheap.tspirv.Samplert(i32 %Samp0Index)
  %Samp1 = tail call target("spirv.Sampler") @llvm.spv.resource.handlefromheap.tspirv.Samplert(i32 %Samp1Index)

  %Value0 = tail call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float>
        @llvm.spv.resource.samplelevel.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(
        target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %Texture, target("spirv.Sampler") %Samp0,
        <2 x float> <float 4.000000e-01, float 5.000000e-01>, float -1.000000e+00, <2 x i32> zeroinitializer)

  %BufPtr0 = call noundef align 4 dereferenceable(16) ptr addrspace(11)
        @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t.i32(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %Buf, i32 0)

  store <4 x float> %Value0, ptr addrspace(11) %BufPtr0, align 4

  %Value1 = tail call reassoc nnan ninf nsz arcp afn noundef nofpclass(nan inf) <4 x float>
        @llvm.spv.resource.samplelevel.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(
        target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %Texture, target("spirv.Sampler") %Samp1,
        <2 x float> <float 4.000000e-01, float 5.000000e-01>, float -1.000000e+00, <2 x i32> zeroinitializer)

  %BufPtr1 = call noundef align 4 dereferenceable(16) ptr addrspace(11)
        @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t.i32(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %Buf, i32 1)
        
  store <4 x float> %Value1, ptr addrspace(11) %BufPtr1, align 4

  ret void
}
