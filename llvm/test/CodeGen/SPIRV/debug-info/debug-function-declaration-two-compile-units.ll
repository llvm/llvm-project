; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Module with two compile units (first.c and second.c). The forward declaration
; names first.c as its file but is not linked to either compile unit in the
; metadata. Both files get DebugCompilationUnit; the declaration is emitted with
; first.c as its source and parented under that compile unit.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH0:%[0-9]+]] = OpString "{{[/\\]}}tmp{{[/\\]}}first.c"
; CHECK-DAG: [[PATH1:%[0-9]+]] = OpString "{{[/\\]}}tmp{{[/\\]}}second.c"
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "fwd_decl"
; CHECK-DAG: [[C100:%[0-9]+]] = OpConstant [[I32]] 100
; CHECK-DAG: [[C5:%[0-9]+]] = OpConstant [[I32]] 5
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32]] 0
; CHECK-DAG: [[C10:%[0-9]+]] = OpConstant [[I32]] 10
; CHECK-DAG: [[C128:%[0-9]+]] = OpConstant [[I32]] 128
; CHECK-DAG: [[DS0:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH0]]
; CHECK-DAG: [[CU0:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS0]] [[C0]]
; CHECK-DAG: [[DS1:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH1]]
; CHECK-DAG: [[CU1:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugCompilationUnit [[C100]] [[C5]] [[DS1]] [[C0]]
; CHECK-DAG: [[TF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeFunction [[C0]] [[VOID]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugFunctionDeclaration [[NAME]] [[TF]] [[DS0]] [[C10]] [[C0]] [[CU0]] [[NAME]] [[C128]]

target triple = "spirv64-unknown-unknown"

define spir_func void @defined() !dbg !9 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0, !20}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None, retainedTypes: !10)
!1 = !DIFile(filename: "first.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{}
!10 = !{!11}
!11 = !DISubprogram(name: "fwd_decl", linkageName: "fwd_decl", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, flags: DIFlagPrototyped, spFlags: 0)

!20 = distinct !DICompileUnit(language: DW_LANG_C99, file: !21, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!21 = !DIFile(filename: "second.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "11111111111111111111111111111111")

!9 = distinct !DISubprogram(name: "defined", linkageName: "defined", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
