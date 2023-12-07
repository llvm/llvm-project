; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/pr26037.ll -o %t2.bc
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s

; CHECK: ![[CU_MAIN:[0-9]+]] = distinct !DICompileUnit(
; CHECK: ![[CU:[0-9]+]] = distinct !DICompileUnit(
; CHECK: !DIImportedEntity({{.*}}, scope: ![[CU]], entity: ![[A:[0-9]+]]
; CHECK: ![[A]] = distinct !DISubprogram(name: "a"
; CHECK: !DIImportedEntity({{.*}}, scope: ![[CU]], entity: ![[LBD:[0-9]+]]
; CHECK: ![[LBD]] = distinct !DILexicalBlock(scope: ![[D:[0-9]+]]
; CHECK: ![[D]] = distinct !DISubprogram(name: "d"

define void @_ZN1A1aEv() #0 !dbg !4 {
entry:
  ret void, !dbg !14
}

define void @_ZN1A1dEv() #0 !dbg !20 {
entry:
  ret void, !dbg !22
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 256934) (llvm/trunk 256936)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, imports: !9)
!1 = !DIFile(filename: "a2.cc", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEv", scope: !5, file: !1, line: 7, type: !6, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DINamespace(name: "A", scope: null)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!9 = !{!10, !16}
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !4, file: !1, line: 8)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.8.0 (trunk 256934) (llvm/trunk 256936)"}
!14 = !DILocation(line: 7, column: 12, scope: !4)
!16 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !19, file: !1, line: 8)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 10, column: 8)
!20 = distinct !DISubprogram(name: "d", linkageName: "_ZN1A1dEv", scope: !5, file: !1, line: 10, type: !6, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 10, column: 8, scope: !20)
