; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV --implicit-check-not=DebugTypeMember
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Under DWARF 5 a static data member is a DIDerivedType tagged DW_TAG_variable,
; not DW_TAG_member, so the member loop skips it and no DebugTypeMember is
; emitted. The composite is still emitted, with no members. Adding support
; for this is tracked in https://github.com/llvm/llvm-project/issues/211842.

; CHECK-SPIRV: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-SPIRV: OpExtInst {{%[0-9]+}} {{%[0-9]+}} DebugTypeComposite

define spir_func void @test() !dbg !10 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "composite.hlsl", directory: "/src")
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_variable, name: "sm", scope: !5, file: !3, line: 2, baseType: !8, flags: DIFlagStaticMember)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
