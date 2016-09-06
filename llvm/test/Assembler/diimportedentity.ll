; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !3, !4, !5, !5}
!named = !{!0, !3, !4, !5, !6}

!llvm.module.flags = !{!7}
!llvm.dbg.cu = !{!1}

; CHECK:      !0 = distinct !DISubprogram({{.*}})
!0 = distinct !DISubprogram(name: "foo", isDefinition: true, unit: !1)

!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !2,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
; CHECK: !3 = !DICompositeType({{.*}})
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "Class", size: 32, align: 32)

; CHECK-NEXT: !4 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "foo", scope: !0, entity: !1, line: 7)
!4 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "foo", scope: !0,
                       entity: !1, line: 7)

; CHECK-NEXT: !5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0)
!5 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0)
!6 = !DIImportedEntity(tag: DW_TAG_imported_module, name: "", scope: !0, entity: null,
                       line: 0)
!7 = !{i32 2, !"Debug Info Version", i32 3}
