; Test input for ../X86/module-odr-language.test.
;
; The object file: a C compile unit, whose language does not allow ODR
; deduplication, and the gmodules skeleton unit referring to the C++ module.

target triple = "x86_64-apple-darwin"

define void @main() !dbg !100 { ret void }

!llvm.dbg.cu = !{!0, !20}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test",
                            emissionKind: FullDebug, imports: !30)
!1 = !DIFile(filename: "1.c", directory: "")
!100 = distinct !DISubprogram(name: "main", scope: !0, file: !1, line: 1,
                              type: !101, unit: !0,
                              spFlags: DISPFlagDefinition)
!101 = !DISubroutineType(types: !102)
!102 = !{null}
!30 = !{!31}
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0,
                        entity: !32, line: 1)
!32 = !DIModule(scope: !0, name: "M", includePath: ".")

; The skeleton unit for module M: DW_AT_GNU_dwo_name and DW_AT_GNU_dwo_id.
!20 = distinct !DICompileUnit(language: DW_LANG_C99, file: !21,
                              producer: "test", emissionKind: FullDebug,
                              splitDebugFilename: "M.pcm", dwoId: 42)
!21 = !DIFile(filename: "M", directory: "")

!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
