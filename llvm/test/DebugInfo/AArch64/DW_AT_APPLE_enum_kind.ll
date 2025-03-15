; RUN: llc < %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; C++ source to regenerate:
; enum __attribute__((enum_extensibility(open))) OpenEnum {
;   oe1
; } oe;
; 
; enum __attribute__((enum_extensibility(closed))) ClosedEnum {
;   ce1
; } ce;
; 
; $ clang++ -O0 -g debug-info-enum-kind.cpp -c


; CHECK: .debug_abbrev contents:

; CHECK: [3] DW_TAG_enumeration_type DW_CHILDREN_yes
; CHECK: DW_AT_APPLE_enum_kind   DW_FORM_data1

; CHECK: .debug_info contents:

; CHECK: DW_TAG_enumeration_type [3]
; CHECK-DAG: DW_AT_name {{.*}} string = "OpenEnum"
; CHECK-DAG: DW_AT_APPLE_enum_kind [DW_FORM_data1]  (DW_APPLE_ENUM_KIND_Open)

; CHECK: DW_TAG_enumeration_type [3]
; CHECK-DAG: DW_AT_name {{.*}} string = "ClosedEnum"
; CHECK-DAG: DW_AT_APPLE_enum_kind [DW_FORM_data1]  (DW_APPLE_ENUM_KIND_Closed)

source_filename = "enum.cpp"
target triple = "arm64-apple-macosx"

@oe = global i32 0, align 4, !dbg !0
@ce = global i32 0, align 4, !dbg !13

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "oe", scope: !2, file: !3, line: 3, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !12, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "enum.cpp", directory: "/tmp")
!4 = !{!5, !9}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "OpenEnum", file: !3, line: 1, baseType: !6, size: 32, elements: !7, identifier: "_ZTS8OpenEnum", enumKind: DW_APPLE_ENUM_KIND_Open)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{!8}
!8 = !DIEnumerator(name: "oe1", value: 0, isUnsigned: true)
!9 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "ClosedEnum", file: !3, line: 5, baseType: !6, size: 32, elements: !10, identifier: "_ZTS10ClosedEnum", enumKind: DW_APPLE_ENUM_KIND_Closed)
!10 = !{!11}
!11 = !DIEnumerator(name: "ce1", value: 0, isUnsigned: true)
!12 = !{!0, !13}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "ce", scope: !2, file: !3, line: 7, type: !9, isLocal: false, isDefinition: true)
!15 = !{i32 7, !"Dwarf Version", i32 5}
!16 = !{i32 2, !"Debug Info Version", i32 3}
