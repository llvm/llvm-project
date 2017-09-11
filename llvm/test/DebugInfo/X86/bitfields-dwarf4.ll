; RUN: llc -mtriple x86_64-apple-macosx -O0 -filetype=obj -o - %s \
; RUN: | llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: llc -mtriple x86_64-gnu-linux -O0 -filetype=obj -o - %s \
; RUN: | llvm-dwarfdump -debug-info - | FileCheck %s --check-prefix=LINUX
; LINUX-NOT: DW_AT_data_bit_offset
;
; Generated from:
;   #include <stdint.h>
;   #pragma pack(1)
;      struct PackedBits
;      {
;        char a;
;        uint32_t b : 5,
;        c : 27
;      } s;
;   #pragma pack()

source_filename = "bitfield.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

%struct.PackedBits = type <{ i8, i32 }>

@s = common global %struct.PackedBits zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 267633)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "bitfield.c", directory: "/Volumes/Data/llvm")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PackedBits", file: !3, line: 3, size: 40, elements: !7)
!7 = !{!8, !10, !14}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 5, baseType: !9, size: 8)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_data_bit_offset
; CHECK:      DW_AT_data_member_location [DW_FORM_data1]	(0x00)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, line: 6, baseType: !11, size: 5, offset: 8)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !12, line: 183, baseType: !13)
!12 = !DIFile(filename: "/Volumes/Data/llvm/_build.ninja.release/bin/../lib/clang/3.9.0/include/stdint.h", directory: "/Volumes/Data/llvm")
!13 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"b"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK:      DW_AT_bit_size             [DW_FORM_data1]	(0x05)
; CHECK-NOT:  DW_AT_byte_size
; CHECK-NEXT: DW_AT_data_bit_offset      [DW_FORM_data1]	(0x08)
; CHECK-NOT:  DW_AT_data_member_location
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !6, file: !3, line: 7, baseType: !11, size: 27, offset: 13)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"PIC Level", i32 2}
; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"c"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK:      DW_AT_bit_size             [DW_FORM_data1]	(0x1b)
; CHECK-NEXT: DW_AT_data_bit_offset      [DW_FORM_data1]	(0x0d)
; CHECK-NOT:  DW_AT_data_member_location
; CHECK: DW_TAG
!18 = !{!"clang version 3.9.0 (trunk 267633)"}

