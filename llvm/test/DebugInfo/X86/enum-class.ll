; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

source_filename = "test/DebugInfo/X86/enum-class.ll"

@a = global i32 0, align 4, !dbg !0
@b = global i64 0, align 8, !dbg !7
@c = global i32 0, align 4, !dbg !13

!llvm.dbg.cu = !{!18}
!llvm.module.flags = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.cpp", directory: "/Users/echristo/tmp")
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "A", file: !2, line: 1, baseType: !4, size: 32, flags: DIFlagFixedEnum, align: 32, elements: !5)
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = !{!6}
!6 = !DIEnumerator(name: "A1", value: 1)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression()) ; [ DW_TAG_enumerator ]
!8 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 5, type: !9, isLocal: false, isDefinition: true)
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "B", file: !2, line: 2, baseType: !10, size: 64, flags: DIFlagFixedEnum, align: 64, elements: !11)
!10 = !DIBasicType(name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!11 = !{!12}
!12 = !DIEnumerator(name: "B1", value: 1) ; [ DW_TAG_enumerator ]
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = !DIGlobalVariable(name: "c", scope: null, file: !2, line: 6, type: !15, isLocal: false, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "C", file: !2, line: 3, size: 32, align: 32, elements: !16)
!16 = !{!17}
!17 = !DIEnumerator(name: "C1", value: 1) ; [ DW_TAG_enumerator ]
!18 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.2 (trunk 157269) (llvm/trunk 157264)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !19, retainedTypes: !20, globals: !21, imports: !20)
!19 = !{!3, !9, !15}
!20 = !{}
!21 = !{!0, !7, !13}
!22 = !{i32 1, !"Debug Info Version", i32 3}

; CHECK: DW_TAG_enumeration_type [{{.*}}]
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "A")

; CHECK: DW_TAG_enumeration_type [{{.*}}] *
; CHECK: DW_AT_type [DW_FORM_ref4]
; CHECK: DW_AT_enum_class [DW_FORM_flag_present] (true)
; CHECK: DW_AT_name [DW_FORM_strp]          ( .debug_str[{{.*}}] = "B")

; CHECK: DW_TAG_enumeration_type
; CHECK-NOT: DW_AT_enum_class
; CHECK: DW_AT_name [DW_FORM_strp]      ( .debug_str[{{.*}}] = "C")
