; REQUIRES: object-emission
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj < %s | llvm-dwarfdump -v - | \
; RUN:    FileCheck --check-prefix=DEFAULT --check-prefix=BOTH %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -generate-type-units < %s |  \
; RUN:    llvm-dwarfdump -v - | FileCheck --check-prefix=TYPEUNITS --check-prefix=BOTH %s
;
; Check that we generate the DWARF v5 string offsets section correctly when we
; have multiple compile and type units. All units share one contribution to
; the string offsets section.
;
; Constructed from the following sources with
; clang -gdwarf-5 -emit-llvm -S a.cpp
; clang -gdwarf-5 -emit-llvm -S b.cpp
; clang -gdwarf-5 -emit-llvm -S c.cpp
; llvm-link a.ll b.ll c.ll -o test.bc
; llvm-dis test.bc -o test.ll
;
; a.cpp:
; enum E1 {a, b, c};
; E1 glob1;
;
; b.cpp:
; enum E2 {d, e, f};
; E2 glob2;
;
; c.cpp:
; enum E3 {g, h, i};
; E3 glob3;
;
; Check that all 3 compile units have the correct DW_AT_str_offsets_base attributes
; with the correct offsets. Check that strings referenced by compile units 2 and 3
; are displayed correctly.
;
; CU 1
; BOTH:        .debug_info contents:
; BOTH-NOT:    .contents:
; BOTH:        DW_TAG_compile_unit
; BOTH-NOT:    {{DW_TAG|NULL}}
; BOTH:        DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x[[CU1_STROFF:[0-9a-f]+]])
;
; CU 2
; BOTH-NOT:    contents:
; BOTH:        DW_TAG_compile_unit
; BOTH-NOT:    {{DW_TAG|NULL}}
; BOTH:        DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x[[CU1_STROFF]])
; BOTH-NOT:    NULL
; BOTH:        DW_TAG_variable
; BOTH-NOT:    {{DW_TAG|NULL}}
; BOTH:        DW_AT_name [DW_FORM_strx1] ( indexed (00000009) string = "glob2")
;
; CU 3
; BOTH-NOT:    contents:
; BOTH:        DW_TAG_compile_unit
; BOTH-NOT:    {{DW_TAG|NULL}}
; BOTH:        DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x[[CU1_STROFF]])
; BOTH-NOT:    NULL
; BOTH:        DW_TAG_variable
; BOTH-NOT:    {{DW_TAG|NULL}}
; BOTH:        DW_AT_name [DW_FORM_strx1] ( indexed (0000000f) string = "glob3")
;
; Verify that all 3 type units have the proper DW_AT_str_offsets_base attribute.
; TYPEUNITS:      .debug_types contents:
; TYPEUNITS-NOT:  contents:
; TYPEUNITS:      DW_TAG_type_unit
; TYPEUNITS-NOT:  {{DW_TAG|NULL}}
; TYPEUNITS:      DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x[[CU1_STROFF]])
; TYPEUNITS-NOT:  NULL
; TYPEUNITS:      DW_TAG_enumerator
; TYPEUNITS-NOT:  NULL
; TYPEUNITS:      DW_TAG_enumerator
; TYPEUNITS-NOT:  {{DW_TAG|NULL}}
; TYPEUNITS:      DW_AT_name [DW_FORM_strx1] ( indexed (00000005) string = "b")
; TYPEUNITS-NOT:  contents:
; TYPEUNITS:      DW_TAG_type_unit
; TYPEUNITS-NOT:  {{DW_TAG|NULL}}
; TYPEUNITS:      DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x[[CU1_STROFF]])
; TYPEUNITS-NOT:  NULL
; TYPEUNITS:      DW_TAG_enumeration_type
; TYPEUNITS:      DW_AT_name [DW_FORM_strx1] ( indexed (0000000d) string = "E2")
; TYPEUNITS-NOT:  contents:
; TYPEUNITS:      DW_TAG_type_unit
; TYPEUNITS-NOT:  {{DW_TAG|NULL}}
; TYPEUNITS:      DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x[[CU1_STROFF]])
; TYPEUNITS-NOT:  NULL
; TYPEUNITS:      DW_TAG_enumeration_type
; TYPEUNITS:      DW_AT_name [DW_FORM_strx1] ( indexed (00000013) string = "E3")
;
; Extract the offset of a string to verify that it is referenced in the string
; offsets section.
; BOTH:           .debug_str contents:
; BOTH-NOT:       contents:
; BOTH:           0x[[GLOB2OFF:[0-9a-f]+]]: "glob2"
;
; Check the .debug_str_offsets section header and make sure the referenced string
; has the correct offset.
; BOTH:           .debug_str_offsets contents:
; BOTH-NEXT:      0x00000000: Contribution size = 84, Format = DWARF32, Version = 5
; BOTH-NEXT:      0x[[CU1_STROFF]]:
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; BOTH-NEXT:      {{.*:}}
; The string with index 9 should be "glob2"
; BOTH-NEXT:      {{.*:}} [[GLOB2OFF]]
;
; ModuleID = 'test.bc'
source_filename = "llvm-link"

@glob1 = global i32 0, align 4, !dbg !0
@glob2 = global i32 0, align 4, !dbg !11
@glob3 = global i32 0, align 4, !dbg !22

!llvm.dbg.cu = !{!2, !13, !24}
!llvm.ident = !{!33, !33, !33}
!llvm.module.flags = !{!34, !35, !36}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glob1", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 322415)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10)
!3 = !DIFile(filename: "a1.cpp", directory: "/home/test", checksumkind: CSK_MD5, checksum: "2ca3eeed18355d6ebbae671eafda5aae")
!4 = !{!5}
!5 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "E1", file: !3, line: 1, size: 32, elements: !6, identifier: "_ZTS2E1")
!6 = !{!7, !8, !9}
!7 = !DIEnumerator(name: "a", value: 0)
!8 = !DIEnumerator(name: "b", value: 1)
!9 = !DIEnumerator(name: "c", value: 2)
!10 = !{!0}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "glob2", scope: !13, file: !14, line: 2, type: !16, isLocal: false, isDefinition: true)
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "clang version 7.0.0 (trunk 322415)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !15, globals: !21)
!14 = !DIFile(filename: "b.cpp", directory: "/home/test", checksumkind: CSK_MD5, checksum: "0e254f89617ecb6c4e5473546a99435c")
!15 = !{!16}
!16 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "E2", file: !14, line: 1, size: 32, elements: !17, identifier: "_ZTS2E2")
!17 = !{!18, !19, !20}
!18 = !DIEnumerator(name: "d", value: 0)
!19 = !DIEnumerator(name: "e", value: 1)
!20 = !DIEnumerator(name: "f", value: 2)
!21 = !{!11}
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = distinct !DIGlobalVariable(name: "glob3", scope: !24, file: !25, line: 2, type: !27, isLocal: false, isDefinition: true)
!24 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !25, producer: "clang version 7.0.0 (trunk 322415)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !26, globals: !32)
!25 = !DIFile(filename: "c.cpp", directory: "/home/test", checksumkind: CSK_MD5, checksum: "7835aaaa683fa09d295adef0e934d392")
!26 = !{!27}
!27 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "E3", file: !25, line: 1, size: 32, elements: !28, identifier: "_ZTS2E3")
!28 = !{!29, !30, !31}
!29 = !DIEnumerator(name: "g", value: 0)
!30 = !DIEnumerator(name: "h", value: 1)
!31 = !DIEnumerator(name: "i", value: 2)
!32 = !{!22}
!33 = !{!"clang version 7.0.0 (trunk 322415)"}
!34 = !{i32 2, !"Dwarf Version", i32 5}
!35 = !{i32 2, !"Debug Info Version", i32 3}
!36 = !{i32 1, !"wchar_size", i32 4}
