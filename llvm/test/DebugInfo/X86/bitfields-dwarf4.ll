; RUN: llc -mtriple x86_64-apple-macosx -O0 -filetype=obj -o - %s \
; RUN: | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; RUN: llc -mtriple x86_64-gnu-linux -O0 -filetype=obj -o - %s \
; RUN: | llvm-dwarfdump -debug-dump=info - | FileCheck %s --check-prefix=LINUX
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

@s = common global %struct.PackedBits zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 267633)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "bitfield.c", directory: "/Volumes/Data/llvm")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "s", scope: !0, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, variable: %struct.PackedBits* @s)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "PackedBits", file: !1, line: 3, size: 40, align: 8, elements: !6)
!6 = !{!7, !9, !13}

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"a"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_data_bit_offset
; CHECK:      DW_AT_data_member_location [DW_FORM_data1]	(0x00)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !5, file: !1, line: 5, baseType: !8, size: 8, align: 8)

!8 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"b"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK:      DW_AT_bit_size             [DW_FORM_data1]	(0x05)
; CHECK-NOT:  DW_AT_byte_size
; CHECK-NEXT: DW_AT_data_bit_offset      [DW_FORM_data1]	(0x08)
; CHECK-NOT:  DW_AT_data_member_location
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !1, line: 6, baseType: !10, size: 5, align: 32, offset: 8)

!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !11, line: 183, baseType: !12)
!11 = !DIFile(filename: "/Volumes/Data/llvm/_build.ninja.release/bin/../lib/clang/3.9.0/include/stdint.h", directory: "/Volumes/Data/llvm")
!12 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)

; CHECK: DW_TAG_member
; CHECK-NEXT: DW_AT_name{{.*}}"c"
; CHECK-NOT:  DW_TAG
; CHECK-NOT:  DW_AT_bit_offset
; CHECK-NOT:  DW_AT_byte_size
; CHECK:      DW_AT_bit_size             [DW_FORM_data1]	(0x1b)
; CHECK-NEXT: DW_AT_data_bit_offset      [DW_FORM_data1]	(0x0d)
; CHECK-NOT:  DW_AT_data_member_location
; CHECK: DW_TAG
!13 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !5, file: !1, line: 7, baseType: !10, size: 27, align: 32, offset: 13)

!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"PIC Level", i32 2}
!17 = !{!"clang version 3.9.0 (trunk 267633)"}
