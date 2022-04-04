; RUN: llc -mtriple x86_64-apple-macosx -O0 -filetype=obj -o %t_le.o %s
; RUN: llvm-dwarfdump -v -debug-info %t_le.o | FileCheck %s

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
; CHECK-NEXT: DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"b"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x4)

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"c"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_byte_size  {{.*}} (0x04)
; CHECK-NEXT: DW_AT_bit_size   {{.*}} (0x01)
; CHECK-NEXT: DW_AT_bit_offset {{.*}} (0x1f)
; CHECK:      DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x8)

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"d"
; CHECK-NOT: DW_TAG_member
; CHECK:      DW_AT_byte_size  {{.*}} (0x04)
; CHECK-NEXT: DW_AT_bit_size   {{.*}} (0x1c)
; CHECK-NEXT: DW_AT_bit_offset {{.*}} (0x03)
; CHECK-NEXT: DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x8)

; ModuleID = 'bitfields.c'
source_filename = "test/DebugInfo/X86/bitfields.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

%struct.bitfield = type <{ i8, [3 x i8], i64 }>

@b = common global %struct.bitfield zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "bitfields.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "bitfield", file: !3, line: 1, size: 96, elements: !7)
!7 = !{!8, !10, !11, !12}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 2, baseType: !9, size: 2)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, line: 3, baseType: !9, size: 32, offset: 32)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !6, file: !3, line: 4, baseType: !9, size: 1, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !6, file: !3, line: 5, baseType: !9, size: 28, offset: 65)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 240548) (llvm/trunk 240554)"}

