; RUN: not llc -O0 -mtriple=spirv-vulkan-compute %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: SampleErrorsDebug.ll:24:10: Non-constant offsets are not supported in sample instructions.

@.str = private unnamed_addr constant [4 x i8] c"img\00", align 1
@.str.1 = private unnamed_addr constant [5 x i8] c"samp\00", align 1

define <4 x float> @sample_debug(<2 x i32> %offset) !dbg !5 {
entry:
  %img = tail call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32 0, i32 0, i32 1, i32 0, ptr @.str), !dbg !10
  %sampler = tail call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32 0, i32 1, i32 1, i32 0, ptr @.str.1), !dbg !11
  %res = call <4 x float> @llvm.spv.resource.sample.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) %img, target("spirv.Sampler") %sampler, <2 x float> zeroinitializer, <2 x i32> %offset), !dbg !12
  ret <4 x float> %res
}

declare target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding.tspirv.Image_f32_1_2_0_0_1_0t(i32, i32, i32, i32, ptr)
declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.tspirv.Samplert(i32, i32, i32, i32, ptr)
declare <4 x float> @llvm.spv.resource.sample.v4f32.tspirv.Image_f32_1_2_0_0_1_0t.tspirv.Samplert.v2f32.v2i32(target("spirv.Image", float, 1, 2, 0, 0, 1, 0), target("spirv.Sampler"), <2 x float>, <2 x i32>)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "SampleErrorsDebug.ll", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "sample_debug", scope: !1, file: !1, line: 20, type: !6, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 22, column: 10, scope: !5)
!11 = !DILocation(line: 23, column: 14, scope: !5)
!12 = !DILocation(line: 24, column: 10, scope: !5)