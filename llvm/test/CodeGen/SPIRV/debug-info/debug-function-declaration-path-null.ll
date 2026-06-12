; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Forward DISubprogram with file: null and line: 0 (verifier-legal for declarations).
; Empty full path is still lowered: OpString "", DebugSource for that file operand,
; then DebugFunctionDeclaration (same idea as SPIRV-LLVM-Translator getSource).

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "/tmp/path-null.c"
; CHECK-DAG: [[EMPTY:%[0-9]+]] = OpString ""
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "no_file_path"
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0
; CHECK-DAG: [[C128:%[0-9]+]] = OpConstant [[I32]] 128
; CHECK-DAG: [[DS_CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[CU:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS_CU]] [[C0]]
; CHECK-DAG: [[TF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeFunction [[C0]] [[VOID]]
; CHECK-DAG: [[DS_DECL:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[EMPTY]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugFunctionDeclaration [[NAME]] [[TF]] [[DS_DECL]] [[C0]] [[C0]] [[CU]] [[NAME]] [[C128]]

target triple = "spirv64-unknown-unknown"

define spir_func void @defined() !dbg !9 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, retainedTypes: !10)
!1 = !DIFile(filename: "path-null.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{}
!10 = !{!11}

!11 = !DISubprogram(name: "no_file_path", linkageName: "no_file_path", scope: !1, file: null, line: 0, type: !6, scopeLine: 0, flags: DIFlagPrototyped, spFlags: 0)
!9 = distinct !DISubprogram(name: "defined", linkageName: "defined", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
