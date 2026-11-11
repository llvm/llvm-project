; XFAIL: target={{.*}}-aix{{.*}}, target={{.*}}-zos{{.*}}
; RUN: %llc_dwarf -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck --check-prefix=NAMES %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; A type's name may be encoded in the type's specification. Test that it's indexed
; under the specification's name.

; CHECK: DW_TAG_structure_type
; CHECK-NOT: DW_AT_name
; CHECK:   DW_AT_specification ([[DECL:0x[0-9a-f]+]] "Specified")
; CHECK: [[DECL]]: DW_TAG_structure_type
; CHECK:   DW_AT_name ("Specified")

; "Specified" is indexed once for the type that has the name and once for the
; type that inherits it.
; NAMES: Name count: 2
; NAMES: String: {{.*}} "Specified"
; NAMES-NEXT: Entry @ {{.*}} {
; NAMES-NEXT:   Abbrev: {{.*}}
; NAMES-NEXT:   Tag: DW_TAG_structure_type
; NAMES-NEXT:   DW_IDX_die_offset: {{0x[0-9a-f]+}}
; NAMES-NEXT:   DW_IDX_parent: {{.*}}
; NAMES-NEXT: }
; NAMES-NEXT: Entry @ {{.*}} {
; NAMES-NEXT:   Abbrev: {{.*}}
; NAMES-NEXT:   Tag: DW_TAG_structure_type
; NAMES-NEXT:   DW_IDX_die_offset: {{0x[0-9a-f]+}}

; VERIFY: No errors.

@var = global i8 0, !dbg !10

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug, globals: !5)
!1 = !DIFile(filename: "a.cpp", directory: "/")
!5 = !{!10}
!6 = !{i32 2, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "Specified", file: !1, line: 1, size: 8)
; The name is inherited from !8.
!9 = !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 1, size: 8, specification: !8)

!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "var", scope: !0, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true)
