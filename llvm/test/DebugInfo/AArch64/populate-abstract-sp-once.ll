; Check that abstract DIEs for inlined subprograms and lexical scopes
; are populated only once.

; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump - -o - | FileCheck --implicit-check-not=DW_TAG_lexical_scope --implicit-check-not DW_TAG_subprogram %s

; CHECK:  DW_TAG_compile_unit
; CHECK:    DW_TAG_namespace
; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_declaration (true)
; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_declaration (true)
; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_declaration (true)
; CHECK:      NULL

; CHECK:  [[ABSTRACT_SP:0x[0-9a-f]+]]: DW_TAG_subprogram
; CHECK:    DW_AT_inline (DW_INL_inlined)

; CHECK:    DW_TAG_lexical_block
; CHECK:      DW_TAG_imported_module
; CHECK:      NULL

; CHECK:    NULL

; CHECK:  DW_TAG_subprogram
; CHECK:    DW_TAG_inlined_subroutine
; CHECK:      DW_AT_abstract_origin ([[ABSTRACT_SP]]
; CHECK:    NULL
; CHECK:  DW_TAG_subprogram
; CHECK:    DW_TAG_inlined_subroutine
; CHECK:      DW_AT_abstract_origin ([[ABSTRACT_SP]]
; CHECK:    NULL

target triple = "aarch64-unknown-linux-gnu"

define void @_ZN12_GLOBAL__N_117MapRegionCounters14TraverseIfStmtEPN5clang6IfStmtE() !dbg !4 {
entry:
  ret void, !dbg !8
}

define void @_ZN12_GLOBAL__N_117MapRegionCounters9VisitStmtEPN5clang4StmtE() !dbg !15 {
entry:
  ret void, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "CodeGenPGO.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "TraverseIfStmt", linkageName: "_ZN12_GLOBAL__N_117MapRegionCounters14TraverseIfStmtEPN5clang6IfStmtE", scope: !5, file: !1, line: 364, type: !6, scopeLine: 364, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !7, retainedNodes: !2, keyInstructions: true)
!5 = !DINamespace(name: "llvm", scope: null)
!6 = distinct !DISubroutineType(types: !2)
!7 = !DISubprogram(name: "TraverseIfStmt", linkageName: "_ZN12_GLOBAL__N_117MapRegionCounters14TraverseIfStmtEPN5clang6IfStmtE", scope: !5, file: !1, line: 364, type: !6, scopeLine: 364, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized)
!8 = !DILocation(line: 982, column: 39, scope: !9, inlinedAt: !14, atomGroup: 6, atomRank: 2)
!9 = distinct !DISubprogram(name: "combine", linkageName: "_ZN12_GLOBAL__N_17PGOHash7combineENS0_8HashTypeE", scope: !5, file: !1, line: 966, type: !6, scopeLine: 966, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !10, retainedNodes: !11, keyInstructions: true)
!10 = !DISubprogram(name: "combine", linkageName: "_ZN12_GLOBAL__N_17PGOHash7combineENS0_8HashTypeE", scope: !5, file: !1, line: 140, type: !6, scopeLine: 140, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized)
!11 = !{!12}
!12 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !13, entity: !5, file: !1, line: 973)
!13 = distinct !DILexicalBlock(scope: !9, file: !1, line: 972, column: 7)
!14 = distinct !DILocation(line: 393, column: 10, scope: !4)
!15 = distinct !DISubprogram(name: "VisitStmt", linkageName: "_ZN12_GLOBAL__N_117MapRegionCounters9VisitStmtEPN5clang4StmtE", scope: !5, file: !1, line: 355, type: !6, scopeLine: 355, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !16, retainedNodes: !2, keyInstructions: true)
!16 = !DISubprogram(name: "VisitStmt", linkageName: "_ZN12_GLOBAL__N_117MapRegionCounters9VisitStmtEPN5clang4StmtE", scope: !5, file: !1, line: 355, type: !6, scopeLine: 355, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized)
!17 = !DILocation(line: 982, column: 13, scope: !9, inlinedAt: !18)
!18 = distinct !DILocation(line: 360, column: 12, scope: !15)
