; REQUIRES: object-emission
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj < %s | llvm-dwarfdump -v - \
; RUN:   | FileCheck --check-prefix=MONOLITHIC %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -split-dwarf-file=foo.dwo -filetype=obj < %s \
; RUN:   | llvm-dwarfdump -v - | FileCheck --check-prefix=SPLIT %s

; This basic test checks the emission of a DWARF v5 string offsets table in
; the split and non-split (monolithic) scenario.
;
; Constructed from the following source with
; clang -S -emit-llvm -gdwarf-5
;
; enum E {a, b, c};
; E glob;
;
; In the non-split scenario, check the DW_AT_str_offsets_base attribute
; in .debug_abbrev.
; MONOLITHIC:          .debug_abbrev contents:
; MONOLITHIC-NOT:      contents:
; MONOLITHIC:          DW_TAG_compile_unit
; MONOLITHIC-NOT:      DW_TAG
; MONOLITHIC:          DW_AT_str_offsets_base DW_FORM_sec_offset

; Check that indexed strings come out correctly and that the DW_str_offsets_base attribute
; is there and has the right value.
; MONOLITHIC:          .debug_info contents:
; MONOLITHIC-NOT:      contents:
; MONOLITHIC:          DW_TAG_compile_unit
; MONOLITHIC-NOT:      {{DW_TAG|NULL}}
; MONOLITHIC:          DW_AT_producer [DW_FORM_strx1] (indexed (00000000) string = "clang{{.*}}")
; MONOLITHIC-NOT:      {{DW_TAG|NULL}}
; MONOLITHIC:          DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x00000008)
; MONOLITHIC-NOT:      {{DW_TAG|NULL}}
; MONOLITHIC:          DW_AT_comp_dir [DW_FORM_strx1] (indexed (00000002) string = "/home/{{.*}}")

; Extract the string offsets from the .debug_str section so we can check that 
; they are referenced correctly in the .debug_str_offsets section.
; MONOLITHIC:          .debug_str contents:
; MONOLITHIC-NEXT:     0x00000000:
; MONOLITHIC-NEXT:     0x[[STRING2:[0-9a-f]]]
; MONOLITHIC-NEXT:     0x[[STRING3:[0-9a-f]]]
; MONOLITHIC-NEXT:     0x[[STRING4:[0-9a-f]]]

; Verify that the .debug_str_offsets section is there and that it starts
; with an 8-byte header, followed by offsets into the .debug_str section.
; MONOLITHIC:          .debug_str_offsets contents:
; MONOLITHIC-NEXT:     Contribution size = 36, Format = DWARF32, Version = 5
; MONOLITHIC-NEXT:     0x00000008: 00000000
; MONOLITHIC-NEXT:     0x0000000c: [[STRING2]]
; MONOLITHIC-NEXT:     0x00000010: [[STRING3]]
; MONOLITHIC-NEXT:     0x00000014: [[STRING4]]

; For split dwarf, verify that the skeleton unit has the DW_AT_str_offsets_base
; attribute and that it has the right value.
;
; SPLIT:      .debug_info contents:
; SPLIT-NEXT: 0x00000000: Compile Unit:{{.*}}DW_UT_skeleton
; SPLIT-NOT:  contents:
; SPLIT:      DW_TAG_skeleton_unit
; SPLIT-NOT:  {{DW_TAG|contents:}}
; SPLIT:      DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x00000008)
; SPLIT:      DW_AT_comp_dir [DW_FORM_strx1] (indexed (00000000) string = "/home/test")
; SPLIT:      DW_AT_dwo_name [DW_FORM_strx1] (indexed (00000001) string = "foo.dwo")

; Check for the split CU in .debug_info.dwo.
; SPLIT:      .debug_info.dwo contents:
; SPLIT-NEXT: 0x00000000: Compile Unit:{{.*}}DW_UT_split_compile
; SPLIT-NOT:  contents:
; SPLIT:      DW_TAG_compile_unit
;
; Check that a couple of indexed strings are displayed correctly and that
; they have the right format (DW_FORM_strx1).
; SPLIT-NOT:  contents:
; SPLIT:      DW_TAG_enumerator
; SPLIT-NOT:  {{DW_TAG|NULL}}
; SPLIT:      DW_AT_name [DW_FORM_strx1]    (indexed (00000001) string = "a")
; SPLIT-NOT:  contents:
; SPLIT:      DW_TAG_enumerator
; SPLIT-NOT:  {{DW_TAG|NULL}}
; SPLIT:      DW_AT_name [DW_FORM_strx1]    (indexed (00000002) string = "b")
;
; Extract the string offsets referenced in the main file by the skeleton unit.
; SPLIT:      .debug_str contents:
; SPLIT-NEXT: 0x[[STRHOMETESTSPLIT:[0-9a-f]*]]: "/home/test"
; SPLIT-NEXT: 0x[[STRESPLIT:[0-9a-f]*]]: "E"
; SPLIT-NEXT: 0x[[STRGLOBSPLIT:[0-9a-f]*]]: "glob"
; SPLIT-NEXT: 0x[[STRFOODWOSPLIT:[0-9a-f]*]]: "foo.dwo"
;
; Extract the string offsets referenced in the .dwo file by the split unit.
; SPLIT:      .debug_str.dwo contents:
; SPLIT-NEXT: 0x00000000:{{.*}}
; SPLIT-NEXT: 0x[[STRING2DWO:[0-9a-f]*]]{{.*}}
; SPLIT-NEXT: 0x[[STRING3DWO:[0-9a-f]*]]{{.*}}
;
; Check the string offsets sections in both the main and the .dwo files and
; verify that the extracted string offsets are referenced correctly. The
; sections should contain only the offsets of strings that are actually
; referenced by the debug info.
; SPLIT:      .debug_str_offsets contents:
; SPLIT-NEXT: 0x00000000: Contribution size = 12, Format = DWARF32, Version = 5
; SPLIT-NEXT: 0x00000008: [[STRHOMETESTSPLIT]] "/home/test"
; SPLIT-NEXT: 0x0000000c: [[STRFOODWOSPLIT]] "foo.dwo"
; SPLIT-EMPTY:

; SPLIT:      .debug_str_offsets.dwo contents:
; SPLIT-NEXT: 0x00000000: Contribution size = 36, Format = DWARF32, Version = 5
; SPLIT-NEXT: 0x00000008: 00000000{{.*}}
; SPLIT-NEXT: 0x0000000c: [[STRING2DWO]]{{.*}}
; SPLIT-NEXT: 0x00000010: [[STRING3DWO]]

@glob = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glob", scope: !2, file: !3, line: 3, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 322415)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !10)
!3 = !DIFile(filename: "en.cpp", directory: "/home/test", checksumkind: CSK_MD5, checksum: "d96b2e2d618e550f0ddd0b6a49c98b02")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "E", file: !3, line: 1, size: 32, elements: !6, identifier: "_ZTS1E")
!6 = !{!7, !8, !9}
!7 = !DIEnumerator(name: "a", value: 0)
!8 = !DIEnumerator(name: "b", value: 1)
!9 = !DIEnumerator(name: "c", value: 2)
!10 = !{!0}
!11 = !{i32 2, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 7.0.0 (trunk 322415)"}
