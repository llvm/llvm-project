; RUN: llc -dwarf-version=4 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s --check-prefix=SINGLE-4

; RUN: llc -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo \
; RUN:     -dwarf-version=4 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s --check-prefix=O-4
; RUN: llvm-dwarfdump -v %t.dwo | FileCheck %s --check-prefix=DWO-4

; RUN: llc -dwarf-version=5 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s --check-prefix=SINGLE-5

; RUN: llc -split-dwarf-file=foo.dwo -split-dwarf-output=%t.dwo \
; RUN:     -dwarf-version=5 -generate-type-units \
; RUN:     -filetype=obj -O0 -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -v - | FileCheck %s --check-prefix=O-5
; RUN: llvm-dwarfdump -v %t.dwo | FileCheck %s --check-prefix=DWO-5

; Looking for DWARF headers to be generated correctly.
; There are 8 variants with 5 formats: v4 CU, v4 TU, v5 normal/partial CU,
; v5 skeleton/split CU, v5 normal/split TU.  Some v5 variants differ only
; in the unit_type code, and the skeleton/split CU differs from normal/partial
; by having one extra field (dwo_id).
; (v2 thru v4 CUs are all the same, and TUs were invented in v4,
; so we don't bother checking older versions.)

; Test case built from:
;struct S {
;  int s1;
;};
;
;S s;

; Verify the v4 non-split headers.
; Note that we check the exact offset of the DIEs because that tells us
; the length of the header.
;
; SINGLE-4: .debug_info contents:
; SINGLE-4: 0x00000000: Compile Unit: {{.*}} version = 0x0004 abbr_offset
; SINGLE-4: 0x0000000b: DW_TAG_compile_unit
;
; SINGLE-4: .debug_types contents:
; SINGLE-4: 0x00000000: Type Unit: {{.*}} version = 0x0004 abbr_offset
; SINGLE-4: 0x00000017: DW_TAG_type_unit

; Verify the v4 split headers.
;
; O-4: .debug_info contents:
; O-4: 0x00000000: Compile Unit: {{.*}} version = 0x0004 abbr_offset
; O-4: 0x0000000b: DW_TAG_compile_unit
;
; DWO-4: .debug_info.dwo contents:
; DWO-4: 0x00000000: Compile Unit: {{.*}} version = 0x0004 abbr_offset
; DWO-4: 0x0000000b: DW_TAG_compile_unit
;
; DWO-4: .debug_types.dwo contents:
; DWO-4: 0x00000000: Type Unit: {{.*}} version = 0x0004 abbr_offset
; DWO-4: 0x00000017: DW_TAG_type_unit

; Verify the v5 non-split headers.
;
; SINGLE-5: .debug_info contents:
; SINGLE-5: 0x00000000: Compile Unit: {{.*}} version = 0x0005 unit_type = DW_UT_compile abbr_offset
; SINGLE-5: 0x0000000c: DW_TAG_compile_unit
;
; FIXME: V5 wants type units in .debug_info not .debug_types.
; SINGLE-5: .debug_types contents:
; SINGLE-5: 0x00000000: Type Unit: {{.*}} version = 0x0005 unit_type = DW_UT_type abbr_offset
; SINGLE-5: 0x00000018: DW_TAG_type_unit

; Verify the v5 split headers.
;
; O-5: .debug_info contents:
; O-5: 0x00000000: Compile Unit: {{.*}} version = 0x0005 unit_type = DW_UT_skeleton abbr_offset
; O-5-SAME:        DWO_id = 0x4ed74084f749d96b
; O-5: 0x00000014: DW_TAG_compile_unit
;
; DWO-5: .debug_info.dwo contents:
; DWO-5: 0x00000000: Compile Unit: {{.*}} version = 0x0005 unit_type = DW_UT_split_compile abbr_offset
; DWO-5-SAME:        DWO_id = 0x4ed74084f749d96b
; DWO-5: 0x00000014: DW_TAG_compile_unit
;
; FIXME: V5 wants type units in .debug_info.dwo not .debug_types.dwo.
; DWO-5: .debug_types.dwo contents:
; DWO-5: 0x00000000: Type Unit: {{.*}} version = 0x0005 unit_type = DW_UT_split_type abbr_offset
; DWO-5: 0x00000018: DW_TAG_type_unit


; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32 }

@s = global %struct.S zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 295942)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "t.cpp", directory: "/home/probinson/projects/scratch")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 32, elements: !7, identifier: "_ZTS1S")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "s1", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 5.0.0 (trunk 295942)"}
