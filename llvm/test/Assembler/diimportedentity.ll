; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !3}
!named = !{!0, !1, !2, !3, !4}

!llvm.module.flags = !{!7}
!llvm.dbg.cu = !{!5}

; CHECK:      !0 = distinct !DISubprogram({{.*}})
; CHECK-NEXT: !1 = !DICompositeType({{.*}})
!0 = distinct !DISubprogram(name: "foo")
!1 = !DICompositeType(tag: DW_TAG_structure_type, name: "Class", size: 32, align: 32)

; CHECK-NEXT: !2 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "foo", scope: !0, entity: !1, line: 7)
!2 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "foo", scope: !0,
                       entity: !1, line: 7)

; CHECK-NEXT: !3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0)
!3 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0)
!4 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "", scope: !0, entity: null,
                       line: 0)

!5 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !6,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2,
                             subprograms: !{!0})
!6 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!7 = !{i32 2, !"Debug Info Version", i32 3}
