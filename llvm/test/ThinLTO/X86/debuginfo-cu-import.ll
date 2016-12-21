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

; Function Attrs: nounwind uwtable
define void @foo() #0 !dbg !31 {
entry:
    ret void, !dbg !34
}

define void @_ZN1A1aEv() #0 !dbg !4 {
entry:
  ret void, !dbg !14
}

define internal void @_ZN1A1bEv() #0 !dbg !8 {
entry:
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0 (trunk 286863) (llvm/trunk 286875)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, macros: !26, retainedTypes: !35, globals: !37, imports: !9)
!1 = !DIFile(filename: "a2.cc", directory: "")
!2 = !{!23}
!4 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEv", scope: !5, file: !1, line: 7, type: !6, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !30)
!5 = !DINamespace(name: "A", scope: null, file: !1, line: 1)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "b", linkageName: "_ZN1A1bEv", scope: !5, file: !1, line: 8, type: !6, isLocal: true, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !30)
!9 = !{!10, !16}
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !4, line: 8)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 4.0.0 (trunk 286863) (llvm/trunk 286875)"}
!14 = !DILocation(line: 7, column: 12, scope: !4)
!15 = !DILocation(line: 8, column: 24, scope: !8)
!16 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !19, line: 8)
!17 = distinct !DILexicalBlock(scope: !18, file: !1, line: 9, column: 8)
!18 = distinct !DISubprogram(name: "c", linkageName: "_ZN1A1cEv", scope: !5, file: !1, line: 9, type: !6, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !30)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 10, column: 8)
!20 = distinct !DISubprogram(name: "d", linkageName: "_ZN1A1dEv", scope: !5, file: !1, line: 10, type: !6, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !30)
!21 = !DILocation(line: 9, column: 8, scope: !18)
!22 = !DILocation(line: 10, column: 8, scope: !20)
!23 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "enum1", scope: !5, file: !1, line: 50, size: 32, elements: !30, identifier: "_ZTSN9__gnu_cxx12_Lock_policyE")
!24 = !{!25}
!25 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "StateTag", scope: !5, file: !1, line: 1653, size: 32, elements: !30, identifier: "_ZTSN2v88StateTagE")
!26 = !{!27}
!27 = !DIMacroFile(line: 0, file: !1, nodes: !28)
!28 = !{!29}
!29 = !DIMacro(type: DW_MACINFO_define, line: 3, name: "X", value: "5")
!30 = !{}
!31 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !32, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, variables: !30)
!32 = !DISubroutineType(types: !33)
!33 = !{null}
!34 = !DILocation(line: 3, column: 1, scope: !31)
!35 = !{!36}
!36 = !DICompositeType(tag: DW_TAG_structure_type, name: "Base", line: 1, size: 32, align: 32, file: !1, elements: !30, identifier: "_ZTS4Base")
!37 = !{!38}
!38 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "version", scope: !5, file: !1, line: 2, type: !36, isLocal: false, isDefinition: true))
