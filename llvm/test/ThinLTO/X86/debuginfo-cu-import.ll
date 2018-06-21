; Test to ensure only the necessary DICompileUnit fields are imported
; for ThinLTO

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/debuginfo-cu-import.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc

; Don't import enums, macros, retainedTypes or globals lists.
; Only import local scope imported entities.
; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t.index.bc -o - | llvm-dis -o - | FileCheck %s
; CHECK-NOT: DICompileUnit{{.*}} enums:
; CHECK-NOT: DICompileUnit{{.*}} macros:
; CHECK-NOT: DICompileUnit{{.*}} retainedTypes:
; CHECK-NOT: DICompileUnit{{.*}} globals:
; CHECK: DICompileUnit{{.*}} imports: ![[IMP:[0-9]+]]
; CHECK: ![[IMP]] = !{!{{[0-9]+}}}

; ModuleID = 'debuginfo-cu-import.c'
source_filename = "debuginfo-cu-import.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() !dbg !28 {
entry:
  ret void, !dbg !29
}

define void @_ZN1A1aEv() !dbg !13 {
entry:
  ret void, !dbg !30
}

define internal void @_ZN1A1bEv() !dbg !31 {
entry:
  ret void, !dbg !32
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25, !26}
!llvm.ident = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 286863) (llvm/trunk 286875)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !6, globals: !8, imports: !11, macros: !21)
!1 = !DIFile(filename: "a2.cc", directory: "")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "enum1", scope: !4, file: !1, line: 50, size: 32, elements: !5, identifier: "_ZTSN9__gnu_cxx12_Lock_policyE")
!4 = !DINamespace(name: "A", scope: null)
!5 = !{}
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !1, line: 1, size: 32, align: 32, elements: !5, identifier: "_ZTS4Base")
!8 = !{!9}
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = !DIGlobalVariable(name: "version", scope: !4, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true)
!11 = !{!12, !16}
!12 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !13, file: !1, line: 8)
!13 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEv", scope: !4, file: !1, line: 7, type: !14, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !5)
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !19, file: !1, line: 8)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 9, column: 8)
!18 = distinct !DISubprogram(name: "c", linkageName: "_ZN1A1cEv", scope: !4, file: !1, line: 9, type: !14, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !5)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 10, column: 8)
!20 = distinct !DISubprogram(name: "d", linkageName: "_ZN1A1dEv", scope: !4, file: !1, line: 10, type: !14, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !5)
!21 = !{!22}
!22 = !DIMacroFile(file: !1, nodes: !23)
!23 = !{!24}
!24 = !DIMacro(type: DW_MACINFO_define, line: 3, name: "X", value: "5")
!25 = !{i32 2, !"Dwarf Version", i32 4}
!26 = !{i32 2, !"Debug Info Version", i32 3}
!27 = !{!"clang version 4.0.0 (trunk 286863) (llvm/trunk 286875)"}
!28 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !14, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, variables: !5)
!29 = !DILocation(line: 3, column: 1, scope: !28)
!30 = !DILocation(line: 7, column: 12, scope: !13)
!31 = distinct !DISubprogram(name: "b", linkageName: "_ZN1A1bEv", scope: !4, file: !1, line: 8, type: !14, isLocal: true, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !5)
!32 = !DILocation(line: 8, column: 24, scope: !31)

