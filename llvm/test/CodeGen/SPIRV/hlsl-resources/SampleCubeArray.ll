; RUN: llc -O0 -mtriple=spirv-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; An arrayed cube image needs SampledCubeArray in addition to Shader.
; CHECK-DAG: OpCapability Shader
; CHECK-DAG: OpCapability SampledCubeArray

; CHECK-DAG: %[[float:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[v4float:[0-9]+]] = OpTypeVector %[[float]] 4
; CHECK-DAG: %[[image:[0-9]+]] = OpTypeImage %[[float]] Cube 2 1 0 1 Unknown
; CHECK-DAG: %[[sampled_image:[0-9]+]] = OpTypeSampledImage %[[image]]
; CHECK-DAG: %[[sampler:[0-9]+]] = OpTypeSampler
; CHECK-DAG: %[[lod:[0-9]+]] = OpConstant %[[float]] 0

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"out\00", align 1

define void @main() {
entry:
  %img = tail call target("spirv.Image", float, 3, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_3_2_1_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)

; The coordinate is a 4-vector: xyz select the face direction, w the cube index.
; CHECK: %[[img_val:[0-9]+]] = OpLoad %[[image]]
; CHECK: %[[sampler_val:[0-9]+]] = OpLoad %[[sampler]]
; CHECK: %[[si:[0-9]+]] = OpSampledImage %[[sampled_image]] %[[img_val]] %[[sampler_val]]
; CHECK: %[[res:[0-9]+]] = OpImageSampleExplicitLod %[[v4float]] %[[si]] %{{[0-9]+}} Lod %[[lod]]
  %res = call <4 x float> @llvm.spv.resource.samplelevel.v4f32.tspirv.Image_f32_3_2_1_0_1_0t.tspirv.Samplert.v4f32.v3i32(target("spirv.Image", float, 3, 2, 1, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <4 x float> <float 1.0, float 0.0, float 0.0, float 0.0>, float 0.0, <3 x i32> zeroinitializer)

; CHECK: %[[out_handle:[0-9]+]] = OpLoad {{.*}}
; CHECK: OpImageWrite %[[out_handle]] {{.*}}
  %out = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_1t(i32 0, i32 2, i32 1, i32 0, ptr @.str.2)
  %out_ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %out, i32 0)
  store <4 x float> %res, ptr addrspace(11) %out_ptr
  ret void
}

declare target("spirv.Image", float, 3, 2, 1, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_3_2_1_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.samplelevel.v4f32.tspirv.Image_f32_3_2_1_0_1_0t.tspirv.Samplert.v4f32.v3i32(target("spirv.Image", float, 3, 2, 1, 0, 1, 0), target("spirv.Sampler"), <4 x float>, float, <3 x i32>)
declare target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_5_2_0_0_2_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1), i32)
