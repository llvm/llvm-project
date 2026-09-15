; RUN: not llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; NonSemantic.Shader.DebugInfo encodes Size and Offset as 32-bit OpConstants,
; so a wider value is rejected instead of silently truncated.

; CHECK: SPIR-V debug info: size of composite type 'Wide' does not fit in 32 bits: 8589934592

define spir_func void @test() !dbg !9 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "wide.hlsl", directory: "/src")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "Wide", file: !3, line: 1, size: 8589934592, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", file: !3, line: 2, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 10, type: !10, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
