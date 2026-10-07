; Test input for ../X86/module-import-canonical-duplicate.test.
;
; The first object file of the link, built against the copy of M under a/.

target triple = "x86_64-apple-darwin"

define void @main() !dbg !100 { ret void }

!llvm.dbg.cu = !{!0, !20}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1,
                            producer: "test", emissionKind: FullDebug,
                            imports: !30)
!1 = !DIFile(filename: "dup1.m", directory: "")
!100 = distinct !DISubprogram(name: "main", scope: !0, file: !1, line: 1,
                              type: !101, unit: !0,
                              spFlags: DISPFlagDefinition)
!101 = !DISubroutineType(types: !102)
!102 = !{null}
!30 = !{!31}
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0,
                        entity: !32, line: 1)
!32 = !DIModule(scope: !0, name: "M", includePath: "/tmp/importer1")

; The skeleton unit referencing a/M.pcm. Its dwo id has to match dup-a.ll's.
!20 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !21,
                              producer: "test", emissionKind: FullDebug,
                              splitDebugFilename: "a/M.pcm", dwoId: 42)
!21 = !DIFile(filename: "M", directory: "")

!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
