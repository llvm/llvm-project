; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Two distinct !DIBasicType nodes named "int" share one OpString "int".

; CHECK: [[ext:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-COUNT-1: OpString "int"
; CHECK-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[type_int32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[flag_zero:%[0-9]+]] = OpConstant [[type_int32]] 0{{$}}
; CHECK-DAG: OpExtInst [[type_void]] [[ext]] DebugTypeFunction [[flag_zero]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @dedupe_a() !dbg !10 {
entry:
  ret i32 0
}

define spir_func i32 @dedupe_b() !dbg !11 {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version XX.X", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-type-function-int-string-dedup.c", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{!8}
!8 = distinct !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!9 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !12)
!12 = !{!13}
!13 = distinct !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!10 = distinct !DISubprogram(name: "dedupe_a", linkageName: "dedupe_a", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = distinct !DISubprogram(name: "dedupe_b", linkageName: "dedupe_b", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
