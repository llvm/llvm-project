; Test input for ../X86/module-import-canonical-priority.test.
;
; Like 1.ll, but in an ODR language and without a skeleton unit for M.pcm, so
; this object file contributes a DW_TAG_module for M to the type table without
; owning the unit which describes M in full. It does own N.pcm, which puts the
; module units of the link in a different order than the object files.

target triple = "x86_64-apple-darwin"

define void @main3() !dbg !100 { ret void }

!llvm.dbg.cu = !{!0, !40}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC_plus_plus, file: !1,
                            producer: "test", emissionKind: FullDebug,
                            imports: !30)
!1 = !DIFile(filename: "3.mm", directory: "")
!100 = distinct !DISubprogram(name: "main3", scope: !0, file: !1, line: 1,
                              type: !101, unit: !0,
                              spFlags: DISPFlagDefinition)
!101 = !DISubroutineType(types: !102)
!102 = !{null}
!30 = !{!31}
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0,
                        entity: !32, line: 1)
!32 = !DIModule(scope: !0, name: "M",
                includePath: "/tmp/M.framework/Modules/M.swiftmodule")

; The skeleton unit referencing N.pcm. Its dwo id has to match N-odr.ll's.
!40 = distinct !DICompileUnit(language: DW_LANG_ObjC_plus_plus, file: !41,
                              producer: "test", emissionKind: FullDebug,
                              splitDebugFilename: "N.pcm", dwoId: 43)
!41 = !DIFile(filename: "N", directory: "")

!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
