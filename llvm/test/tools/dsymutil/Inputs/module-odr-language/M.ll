; Test input for ../X86/module-odr-language.test.
;
; Stands in for a Clang module built as C++: one compile unit whose language
; is an ODR language, holding the module's type definitions, a forward
; declaration and a member function declaration under DW_TAG_module.

target triple = "x86_64-apple-darwin"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1,
                            producer: "test", emissionKind: FullDebug,
                            retainedTypes: !2, dwoId: 42)
!1 = !DIFile(filename: "M", directory: "")
!2 = !{!3, !30}
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !4,
                      file: !1, line: 1, size: 32, elements: !5,
                      identifier: "_ZTS1S")
!4 = !DIModule(scope: null, name: "M", includePath: ".")
!5 = !{!6, !20}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !3, file: !1,
                    line: 2, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!20 = !DISubprogram(name: "get", linkageName: "_ZN1S3getEv", scope: !3,
                    file: !1, line: 3, type: !21, scopeLine: 3,
                    flags: DIFlagPrototyped, spFlags: 0)
!21 = !DISubroutineType(types: !22)
!22 = !{!7, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !4,
                       file: !1, line: 5, flags: DIFlagFwdDecl,
                       identifier: "_ZTS1T")
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
