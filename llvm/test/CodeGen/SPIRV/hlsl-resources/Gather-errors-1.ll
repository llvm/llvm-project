; RUN: not llc -O0 -mtriple=spirv-vulkan-compute %s -o - 2>&1 | FileCheck %s

; CHECK: Gather operations with offset are not supported for Cube images.
; ERR-3D: Gather operations are only supported for 2D, Cube, and Rect images.

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1

define void @cube_with_offset() {
entry:
  %img = tail call target("spirv.Image", float, 3, 0, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_3_0_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1)
  %res1 = call <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_3_0_0_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 3, 0, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <3 x float> zeroinitializer, i32 1, <2 x i32> <i32 1, i32 1>)
  ret void
}

declare target("spirv.Image", float, 3, 0, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_3_0_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.gather.v4f32.tspirv.Image_f32_3_0_0_0_1_0t.tspirv.Samplert.v3f32.i32.v2i32(target("spirv.Image", float, 3, 0, 0, 0, 1, 0), target("spirv.Sampler"), <3 x float>, i32, <2 x i32>)
