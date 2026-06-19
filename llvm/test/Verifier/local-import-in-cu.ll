; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: function-local imports are not allowed in a DICompileUnit's imported entities list
; CHECK: warning: ignoring invalid debug info

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, imports: !5)
!2 = !DIFile(filename: "foo.c", directory: "")
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !2, line: 1, unit: !1, type: !8)
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !3, entity: !6)
!5 = !{!4}
!6 = !DINamespace(name: "M", scope: null)
!7 = !{null}
!8 = !DISubroutineType(types: !7)
