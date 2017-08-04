; RUN:  llvm-dis < %s.bc | FileCheck %s
; RUN:  verify-uselistorder < %s.bc

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 308185) (llvm/trunk 308186)", emissionKind: FullDebug, imports: !3)
!1 = !DIFile(filename: "using.ii", directory: "/")
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !8, line: 301)
; CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !5)
!5 = !DINamespace(name: "M", scope: null)
!8 = !DINamespace(name: "N", scope: null)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
