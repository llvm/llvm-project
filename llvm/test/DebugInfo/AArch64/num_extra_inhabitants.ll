; RUN: llc %s -filetype=obj -mtriple arm64e-apple-darwin -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_TAG_base_type
; CHECK: DW_AT_name ("ExtraInhabitantBasicType")
; CHECK: DW_AT_LLVM_num_extra_inhabitants (0xfe)

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("ExtraInhabitantCompositeType")
; CHECK: DW_AT_LLVM_num_extra_inhabitants (0x42)

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name ("NoExtraInhabitantType")
; CHECK-NOT: DW_AT_LLVM_num_extra_inhabitants

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@p = common global i8* null, align 8, !dbg !100
@q = common global i8* null, align 8, !dbg !102
@r = common global i8* null, align 8, !dbg !105

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "/tmp/p.c", directory: "/")
!4 = !{}
!5 = !{!100, !102, !105}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!10 = !DIBasicType(name: "ExtraInhabitantBasicType", size: 1, encoding: DW_ATE_unsigned, num_extra_inhabitants: 254)
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "ExtraInhabitantCompositeType", file: !3, size: 64, num_extra_inhabitants: 66, identifier: "MangledExtraInhabitantCompositeType")
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "NoExtraInhabitantType", file: !3, size: 64, identifier: "MangledNoExtraInhabitantType")

!100 = !DIGlobalVariableExpression(var: !101, expr: !DIExpression())
!101 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 1, type: !10, isLocal: false, isDefinition: true)
!102 = !DIGlobalVariableExpression(var: !103, expr: !DIExpression())
!103 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 1, type: !11, isLocal: false, isDefinition: true)
!104 = distinct !DIGlobalVariable(name: "r", scope: !2, file: !3, line: 1, type: !12, isLocal: false, isDefinition: true)
!105 = !DIGlobalVariableExpression(var: !104, expr: !DIExpression())
