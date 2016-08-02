; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

@a = global i32 0, align 4
@b = global i64 0, align 8
@c = global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.2 (trunk 157269) (llvm/trunk 157264)", isOptimized: false, emissionKind: 0, file: !22, enums: !1, retainedTypes: !15, subprograms: !15, globals: !17, imports:  !15)
!1 = !{!3, !8, !12}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "A", line: 1, size: 32, align: 32, file: !4, baseType: !5, elements: !6)
!4 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = !DIEnumerator(name: "A1", value: 1) ; [ DW_TAG_enumerator ]
!8 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "B", line: 2, size: 64, align: 64, file: !4, baseType: !9, elements: !10)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !DIEnumerator(name: "B1", value: 1) ; [ DW_TAG_enumerator ]
!12 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "C", line: 3, size: 32, align: 32, file: !4, elements: !13)
!13 = !{!14}
!14 = !DIEnumerator(name: "C1", value: 1) ; [ DW_TAG_enumerator ]
!15 = !{}
!17 = !{!19, !20, !21}
!19 = !DIGlobalVariable(name: "a", line: 4, isLocal: false, isDefinition: true, scope: null, file: !4, type: !3, variable: i32* @a)
!20 = !DIGlobalVariable(name: "b", line: 5, isLocal: false, isDefinition: true, scope: null, file: !4, type: !8, variable: i64* @b)
!21 = !DIGlobalVariable(name: "c", line: 6, isLocal: false, isDefinition: true, scope: null, file: !4, type: !12, variable: i32* @c)
!22 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")

; CHECK: DW_TAG_enumeration_type [{{.*}}]
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "A")

; CHECK: DW_TAG_enumeration_type [{{.*}}] *
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]          ( .debug_str[{{.*}}] = "B")

; CHECK: DW_TAG_enumeration_type [6]
; CHECK-NOT: DW_AT_enum_class
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "C")
!23 = !{i32 1, !"Debug Info Version", i32 3}
