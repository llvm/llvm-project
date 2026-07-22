; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Verify SPIR-V resource intrinsics that use implicit derivatives are marked
; convergent so LLVM does not sink them into divergent control flow.

; CHECK: declare float @llvm.spv.resource.calculate.lod.f32.{{.*}} [[CONVERGENT:#[0-9]+]]
declare float @llvm.spv.resource.calculate.lod.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>)

; CHECK: declare float @llvm.spv.resource.calculate.lod.unclamped.f32.{{.*}} [[CONVERGENT]]
declare float @llvm.spv.resource.calculate.lod.unclamped.f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>)

; CHECK: declare <4 x float> @llvm.spv.resource.sample.v4f32.{{.*}} [[CONVERGENT]]
declare <4 x float> @llvm.spv.resource.sample.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, <2 x i32>)

; CHECK: declare <4 x float> @llvm.spv.resource.sample.clamp.v4f32.{{.*}} [[CONVERGENT]]
declare <4 x float> @llvm.spv.resource.sample.clamp.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, <2 x i32>, float)

; CHECK: declare <4 x float> @llvm.spv.resource.samplebias.v4f32.{{.*}} [[CONVERGENT]]
declare <4 x float> @llvm.spv.resource.samplebias.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, float, <2 x i32>)

; CHECK: declare <4 x float> @llvm.spv.resource.samplebias.clamp.v4f32.{{.*}} [[CONVERGENT]]
declare <4 x float> @llvm.spv.resource.samplebias.clamp.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, float, <2 x i32>, float)

; Explicit-derivative sampling should remain non-convergent.
; CHECK: declare <4 x float> @llvm.spv.resource.samplegrad.v4f32.{{.*}} [[NON_CONVERGENT:#[0-9]+]]
declare <4 x float> @llvm.spv.resource.samplegrad.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v3f32.v2f32.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <3 x float>, <2 x float>, <2 x float>, <2 x i32>)

; CHECK: attributes [[CONVERGENT]] = { convergent nocallback nofree nosync nounwind willreturn memory(read) }
; CHECK: attributes [[NON_CONVERGENT]] = { nocallback nofree nosync nounwind willreturn memory(read) }
