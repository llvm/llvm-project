; RUN: llc %s -filetype=obj -mtriple arm64e-apple-darwin -o - \
; RUN:   | llvm-dwarfdump - | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name	("SmallSpareBitsMaskType")
; CHECK: DW_TAG_variant_part
; CHECK: DW_AT_APPLE_spare_bits_mask	(-1152921504606846970)
; CHECK: DW_AT_bit_offset	(0x38)

; CHECK: DW_TAG_structure_type
; CHECK: DW_AT_name	("BigSpareBitsMaskType")
; CHECK: DW_TAG_variant_part
; CHECK: DW_AT_APPLE_spare_bits_mask	(<0x20> 06 00 00 00 00 00 00 f0 07 00 00 00 00 00 00 f0 07 00 00 00 00 00 00 f0 07 00 00 00 00 00 00 f0 )
; CHECK: DW_AT_bit_offset	(0x38)
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@p = common global i8* null, align 8, !dbg !0
@q = common global i8* null, align 8, !dbg !8

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !3, line: 1, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "/tmp/p.c", directory: "/")
!4 = !{}
!5 = !{!0, !8}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 1, type: !14, isLocal: false, isDefinition: true)

!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "SmallSpareBitsMaskType", file: !3, size: 64, elements: !11, identifier: "SmallSpareBitsMaskType")
!11 = !{!12}

!12 = !DICompositeType(tag: DW_TAG_variant_part, file: !3, line: 3, size: 64, offset: 56, spare_bits_mask: 1152921504606846982, elements: !13)
!13 = !{}

!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "BigSpareBitsMaskType", file: !3, size: 64, elements: !15, num_extra_inhabitants: 66, identifier: "BigSpareBitsMaskType")
!15 = !{!16}

!16 = !DICompositeType(tag: DW_TAG_variant_part, file: !3, line: 3, size: 64, offset: 56, spare_bits_mask: 108555083659983933259422293470276692178088180458183830018690888325426562727942)

