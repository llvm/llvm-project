; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypeArray
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; An array whose element is a composite is dropped. Arrays are emitted before
; composites, so the element has no id yet when emitDebugTypeArray runs and the
; whole DebugTypeArray is skipped. The composite itself is still emitted.
; Emitting the DebugType* nodes in dependency order, tracked in
; https://github.com/llvm/llvm-project/issues/211850, would emit the
; DebugTypeArray and change the expected output of this test.

; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: DebugTypeComposite

define spir_func void @test() !dbg !11 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "array-of-struct.hlsl", directory: "/src")

; S arr[4]. Only the array is retained; S is reached through its element type.
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !6)
!6 = !{!DISubrange(count: 4)}
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !3, line: 2, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
