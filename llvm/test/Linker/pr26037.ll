; RUN: llvm-as %s -o %t.bc
; RUN: llvm-as %p/Inputs/pr26037.ll -o %t2.bc
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s

; CHECK: [[A:![0-9]+]] = distinct !DISubprogram(name: "a"
; CHECK: [[B:![0-9]+]] = distinct !DISubprogram(name: "b"
; CHECK: !DIImportedEntity({{.*}}, scope: [[B]], entity: [[A]]

define void @_ZN1A1aEv() #0 !dbg !4 {
entry:
  ret void, !dbg !14
}

define void @_ZN1A1bEv() #0 !dbg !8 {
entry:
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 256934) (llvm/trunk 256936)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, imports: !9)
!1 = !DIFile(filename: "a2.cc", directory: "")
!2 = !{}
!3 = !{!4, !8}
!4 = distinct !DISubprogram(name: "a", linkageName: "_ZN1A1aEv", scope: !5, file: !1, line: 7, type: !6, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DINamespace(name: "A", scope: null, file: !1, line: 1)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "b", linkageName: "_ZN1A1bEv", scope: !5, file: !1, line: 8, type: !6, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!9 = !{!10}
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !8, entity: !4, line: 8)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.8.0 (trunk 256934) (llvm/trunk 256936)"}
!14 = !DILocation(line: 7, column: 12, scope: !4)
!15 = !DILocation(line: 8, column: 24, scope: !8)
