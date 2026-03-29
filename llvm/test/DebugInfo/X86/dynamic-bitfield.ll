; The use of llvm-dis here tests that round-tripping the IR works
; correctly for the expression case.
; RUN: llvm-as < %s | llvm-dis | llc -mtriple=x86_64 -O0 -filetype=obj -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; A basic test of using a DIExpression for DW_AT_data_bit_offset and
; DW_AT_bit_size.

source_filename = "bitfield.c"

%struct.PackedBits = type <{ i8, i32 }>

@s = common global %struct.PackedBits zeroinitializer, align 1, !dbg !2
@value = common global i32 zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "value", scope: !4, file: !5, line: 8, type: !15, isLocal: false, isDefinition: true)
!2 = distinct !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = !DIGlobalVariable(name: "s", scope: !4, file: !5, line: 8, type: !8, isLocal: false, isDefinition: true)


!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, producer: "clang version 3.9.0 (trunk 267633)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, globals: !7)
!5 = !DIFile(filename: "bitfield.c", directory: "/Volumes/Data/llvm")
!6 = !{}
!7 = !{!0, !2}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PackedBits", file: !5, line: 3, size: 40, elements: !9)
!9 = !{!10, !12, !16, !21}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !8, file: !5, line: 5, baseType: !11, size: 8)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_data_bit_offset
; CHECK:      DW_AT_data_member_location [DW_FORM_data1]	(0x00)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !5, line: 6, baseType: !13, size: !3, offset: !3, flags: DIFlagBitField)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !14, line: 183, baseType: !15)
!14 = !DIFile(filename: "/Volumes/Data/llvm/_build.ninja.release/bin/../lib/clang/3.9.0/include/stdint.h", directory: "/Volumes/Data/llvm")
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"b"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK:      DW_AT_bit_size             [DW_FORM_ref4] ({{.*}})
; CHECK-NEXT: DW_AT_data_bit_offset      [DW_FORM_ref4] ({{.*}})
; CHECK-NOT:  DW_AT_data_member_location
!16 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !8, file: !5, line: 7, baseType: !13, size: !DIExpression(DW_OP_constu, 27), offset: !DIExpression(DW_OP_constu, 13), flags: DIFlagBitField)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"PIC Level", i32 2}
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"c"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK:      DW_AT_bit_size             [DW_FORM_exprloc]	(DW_OP_lit27)
; CHECK-NEXT: DW_AT_data_bit_offset      [DW_FORM_exprloc]	(DW_OP_lit13)
; CHECK-NOT:  DW_AT_data_member_location
!20 = !{!"clang version 3.9.0 (trunk 267633)"}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !8, file: !5, line: 7, baseType: !13, offset: !DIExpression(DW_OP_constu, 15), flags: DIFlagBitField)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"d"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK-NOT:  DW_AT_bit_size
; CHECK:      DW_AT_data_bit_offset      [DW_FORM_exprloc]	(DW_OP_lit15)
; CHECK-NOT:  DW_AT_data_member_location
; CHECK: DW_TAG
