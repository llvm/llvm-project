; Test input for ../X86/module-import-canonical-priority.test.
;
; A second module, in an ODR language like M-odr.ll, so that the link holds more
; than one module unit and they are owned by different object files.

target triple = "x86_64-apple-darwin"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC_plus_plus, file: !1,
                            producer: "test", emissionKind: FullDebug,
                            retainedTypes: !2, dwoId: 43)
!1 = !DIFile(filename: "N", directory: "")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !4,
                      file: !1, line: 1, size: 32, elements: !5)
!4 = !DIModule(scope: null, name: "N", includePath: "/tmp/N.framework")
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "count", scope: !3, file: !1,
                    line: 2, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
