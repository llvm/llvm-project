; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypeMember
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Composites are emitted in a single DebugInfoFinder pass. Outer is discovered
; before Inner, so when Outer is emitted its Inner-typed member has no id yet
; and is dropped. Inner is emitted next, and its basic-type member survives.
; Only one DebugTypeMember is emitted here, for Inner.x. Emitting composites in
; dependency order, would add a second DebugTypeMember for Outer.inner
; (see https://github.com/llvm/llvm-project/issues/211850).

; Both composites are still emitted.
; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: DebugTypeComposite
; CHECK-SPIRV: DebugTypeMember
; CHECK-SPIRV: DebugTypeComposite

define spir_func void @test() !dbg !12 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "composite.hlsl", directory: "/src")

; Only Outer is retained. Inner is reached solely through Outer's member, so the
; finder visits Outer first and Inner second.
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Outer", file: !3, line: 1, size: 32, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "inner", file: !3, line: 2, baseType: !8, size: 32)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Inner", file: !3, line: 5, size: 32, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "x", file: !3, line: 6, baseType: !11, size: 32)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 10, type: !13, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
