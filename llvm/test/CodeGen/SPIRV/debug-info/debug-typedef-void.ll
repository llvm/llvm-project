; RUN: llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; typedef void V: a typedef with no base type uses DebugInfoNone as Base Type.

; CHECK-DAG: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[name_v:%[0-9]+]] = OpString "V"
; CHECK-DAG: [[none:%[0-9]+]] = OpExtInst [[void]] [[ext]] DebugInfoNone
; CHECK: OpExtInst [[void]] [[ext]] DebugTypedef [[name_v]] [[none]]

define spir_func void @test() !dbg !6 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4)
!1 = !{i32 7, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "typedef-void.c", directory: "/src")

!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "V", file: !3, line: 1, baseType: null, scope: !3)
!6 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
