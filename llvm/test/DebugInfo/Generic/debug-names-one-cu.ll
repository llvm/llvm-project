; XFAIL: target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; Check the header
; CHECK: CU count: 1
; CHECK: Local TU count: 0
; CHECK: Foreign TU count: 0
; CHECK: Name count: 3
; CHECK: CU[0]: 0x{{[0-9a-f]*}}

; CHECK: Abbreviation [[ABBREV_STRUCT:0x[0-9a-f]*]] {
; CHECK-NEXT:   Tag: DW_TAG_structure_type
; CHECK-NEXT:   DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT: }

; CHECK: Abbreviation [[ABBREV:0x[0-9a-f]*]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT: DW_IDX_parent: DW_FORM_flag_present
; CHECK-NEXT: }

; The entry for A::B must not have an IDX_Parent, since A is only a forward
; declaration.
; CHECK: String: 0x{{[0-9a-f]*}} "B"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV_STRUCT]]
; CHECK-NEXT: Tag: DW_TAG_structure_type
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}
; CHECK-NEXT: }

; CHECK: String: 0x{{[0-9a-f]*}} "someA_B"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}
; CHECK-NEXT: DW_IDX_parent: <parent not indexed>
; CHECK-NEXT: }

; CHECK: String: 0x{{[0-9a-f]*}} "foobar"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}
; CHECK-NEXT: DW_IDX_parent: <parent not indexed>
; CHECK-NEXT: }

; VERIFY: No errors.

@foobar = common dso_local global ptr null, align 8, !dbg !0
@someA_B = common dso_local global ptr null, align 8, !dbg !18

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foobar", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/tmp/cu1.c", directory: "/tmp")
!4 = !{}
!5 = !{!0, !18}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)"}

!13 = !{}
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", scope: !16, file: !3, line: 3, size: 8, elements: !13, identifier: "type_A::B")
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 8, flags: DIFlagFwdDecl, identifier: "type_A")
!17 = distinct !DIGlobalVariable(name: "someA_B", scope: !2, file: !3, line: 1, type: !15, isLocal: false, isDefinition: true)
!18 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
