; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=SPV_KHR_relaxed_extended_instruction --implicit-check-not=OpExtInstWithForwardRefsKHR
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_KHR_relaxed_extended_instruction --spirv-debug-scope-forward-refs=true -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check that SPV_KHR_relaxed_extended_instruction is not declared.

; CHECK: OpExtension "SPV_KHR_non_semantic_info"
; CHECK: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK: DebugTypeBasic
; CHECK: DebugTypedef

define spir_func void @test() !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "nocycle.c", directory: "/src")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "MyInt", file: !3, line: 1, baseType: !7)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
