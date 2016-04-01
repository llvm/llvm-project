; RUN: llc -mtriple x86_64-apple-macosx -O0 -filetype=obj -o %t_le.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t_le.o | FileCheck %s

; Produced at -O0 from:
; struct bitfield {
;   int a : 2;
;   int b : 32;
;   int c : 1;
;   int d : 28;
; };
; struct bitfield b;

; Note that DWARF 2 counts bit offsets backwards from the high end of
; the storage unit to the high end of the bit field.

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_byte_size  {{.*}} (0x04)
; CHECK-NEXT: DW_AT_bit_size   {{.*}} (0x02)
; CHECK-NEXT: DW_AT_bit_offset {{.*}} (0x1e)
; CHECK-NEXT: DW_AT_data_member_location {{.*}} 00

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_data_member_location {{.*}} 04

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"c"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_byte_size  {{.*}} (0x04)
; CHECK-NEXT: DW_AT_bit_size   {{.*}} (0x01)
; CHECK-NEXT: DW_AT_bit_offset {{.*}} (0x1f)
; CHECK-NEXT: DW_AT_data_member_location {{.*}} 08

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"d"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_byte_size  {{.*}} (0x04)
; CHECK-NEXT: DW_AT_bit_size   {{.*}} (0x1c)
; CHECK-NEXT: DW_AT_bit_offset {{.*}} (0x03)
; CHECK-NEXT: DW_AT_data_member_location {{.*}} 08

; ModuleID = 'bitfields.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

%struct.bitfield = type <{ i8, [3 x i8], i64 }>

@b = common global %struct.bitfield zeroinitializer, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "bitfields.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "b", scope: !0, file: !5, line: 8, type: !6, isLocal: false, isDefinition: true, variable: %struct.bitfield* @b)
!5 = !DIFile(filename: "bitfields.c", directory: "/")
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "bitfield", file: !5, line: 1, size: 96, align: 32, elements: !7)
!7 = !{!8, !10, !11, !12}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !5, line: 2, baseType: !9, size: 2, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !5, line: 3, baseType: !9, size: 32, align: 32, offset: 32)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !6, file: !5, line: 4, baseType: !9, size: 1, align: 32, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !6, file: !5, line: 5, baseType: !9, size: 28, align: 32, offset: 65)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)"}
