; Generated from tools/clang/test/CodeGen/debug-info-packed-struct.c
; ModuleID = 'llvm/tools/clang/test/CodeGen/debug-info-packed-struct.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

; RUN: %llc_dwarf -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s
; REQUIRES: object-emission

;  // ---------------------------------------------------------------------
;  // Not packed.
;  // ---------------------------------------------------------------------
;  struct size8 {
;    int i : 4;
;    long long l : 60;
;  };
;  struct layout0 {
;    char l0_ofs0;
;    struct size8 l0_ofs8;
;    int l0_ofs16 : 1;
;  } l0;

%struct.layout0 = type { i8, %struct.size8, i8 }
%struct.size8 = type { i64 }
; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout0"
; CHECK:      DW_AT_byte_size {{.*}} (0x18)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l0_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}}00
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l0_ofs8"
; CHECK:          DW_AT_data_member_location {{.*}}08
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l0_ofs16"
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x1f)
; CHECK:          DW_AT_data_member_location {{.*}}10


; // ---------------------------------------------------------------------
; // Implicitly packed.
; // ---------------------------------------------------------------------
; struct size8_anon {
;   int : 4;
;   long long : 60;
; };
; struct layout1 {
;   char l1_ofs0;
;   struct size8_anon l1_ofs1;
;   int l1_ofs9 : 1;
; } l1;

%struct.layout1 = type <{ i8, %struct.size8_anon, i8, [2 x i8] }>
%struct.size8_anon = type { i64 }

; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout1"
; CHECK:      DW_AT_byte_size {{.*}} (0x0c)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l1_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}}00
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l1_ofs1"
; CHECK:          DW_AT_data_member_location {{.*}}01
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l1_ofs9"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x17)
; CHECK:          DW_AT_data_member_location {{.*}}08

; // ---------------------------------------------------------------------
; // Explicitly packed.
; // ---------------------------------------------------------------------
; #pragma pack(1)
; struct size8_pack1 {
;   int i : 4;
;   long long l : 60;
; };
; struct layout2 {
;   char l2_ofs0;
;   struct size8_pack1 l2_ofs1;
;   int l2_ofs9 : 1;
; } l2;
; #pragma pack()

%struct.layout2 = type <{ i8, %struct.size8_pack1, i8 }>
%struct.size8_pack1 = type { i64 }

; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout2"
; CHECK:      DW_AT_byte_size {{.*}} (0x0a)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l2_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}}00
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l2_ofs1"
; CHECK:          DW_AT_data_member_location {{.*}}01
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l2_ofs9"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x17)
; CHECK:          DW_AT_data_member_location {{.*}}08

; // ---------------------------------------------------------------------
; // Explicitly packed with different alignment.
; // ---------------------------------------------------------------------
; #pragma pack(4)
; struct size8_pack4 {
;   int i : 4;
;   long long l : 60;
; };
; struct layout3 {
;   char l3_ofs0;
;   struct size8_pack4 l3_ofs4;
;   int l3_ofs12 : 1;
; } l 3;
; #pragma pack()


%struct.layout3 = type <{ i8, [3 x i8], %struct.size8_pack4, i8, [3 x i8] }>
%struct.size8_pack4 = type { i64 }

; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout3"
; CHECK:      DW_AT_byte_size {{.*}} (0x10)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l3_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}}00
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l3_ofs4"
; CHECK:          DW_AT_data_member_location {{.*}}04
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l3_ofs12"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x1f)
; CHECK:          DW_AT_data_member_location {{.*}}0c

@l0 = common global %struct.layout0 zeroinitializer, align 8, !dbg !4
@l1 = common global %struct.layout1 zeroinitializer, align 4, !dbg !18
@l2 = common global %struct.layout2 zeroinitializer, align 1, !dbg !25
@l3 = common global %struct.layout3 zeroinitializer, align 4, !dbg !35

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!45, !46}
!llvm.ident = !{!47}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 240791) (llvm/trunk 240790)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "/llvm/tools/clang/test/CodeGen/<stdin>", directory: "/llvm/_build.ninja.release")
!2 = !{}
!3 = !{!4, !18, !25, !35}
!4 = !DIGlobalVariable(name: "l0", scope: !0, file: !5, line: 88, type: !6, isLocal: false, isDefinition: true)
!5 = !DIFile(filename: "/llvm/tools/clang/test/CodeGen/debug-info-packed-struct.c", directory: "/llvm/_build.ninja.release")
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout0", file: !5, line: 15, size: 192, align: 64, elements: !7)
!7 = !{!8, !10, !17}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs0", scope: !6, file: !5, line: 16, baseType: !9, size: 8, align: 8)
!9 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs8", scope: !6, file: !5, line: 17, baseType: !11, size: 64, align: 64, offset: 64)
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8", file: !5, line: 11, size: 64, align: 64, elements: !12)
!12 = !{!13, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !11, file: !5, line: 12, baseType: !14, size: 4, align: 32)
!14 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !11, file: !5, line: 13, baseType: !16, size: 60, offset: 4)
!16 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs16", scope: !6, file: !5, line: 18, baseType: !14, size: 1, align: 32, offset: 128)
!18 = !DIGlobalVariable(name: "l1", scope: !0, file: !5, line: 89, type: !19, isLocal: false, isDefinition: true)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout1", file: !5, line: 34, size: 96, align: 32, elements: !20)
!20 = !{!21, !22, !24}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs0", scope: !19, file: !5, line: 35, baseType: !9, size: 8, align: 8)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs1", scope: !19, file: !5, line: 36, baseType: !23, size: 64, align: 8, offset: 8)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8_anon", file: !5, line: 30, size: 64, align: 8, elements: !2)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs9", scope: !19, file: !5, line: 37, baseType: !14, size: 1, align: 32, offset: 72)
!25 = !DIGlobalVariable(name: "l2", scope: !0, file: !5, line: 90, type: !26, isLocal: false, isDefinition: true)
!26 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout2", file: !5, line: 54, size: 80, align: 8, elements: !27)
!27 = !{!28, !29, !34}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs0", scope: !26, file: !5, line: 55, baseType: !9, size: 8, align: 8)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs1", scope: !26, file: !5, line: 56, baseType: !30, size: 64, align: 8, offset: 8)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8_pack1", file: !5, line: 50, size: 64, align: 8, elements: !31)
!31 = !{!32, !33}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !30, file: !5, line: 51, baseType: !14, size: 4, align: 32)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !30, file: !5, line: 52, baseType: !16, size: 60, offset: 4)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs9", scope: !26, file: !5, line: 57, baseType: !14, size: 1, align: 32, offset: 72)
!35 = !DIGlobalVariable(name: "l3", scope: !0, file: !5, line: 91, type: !36, isLocal: false, isDefinition: true)
!36 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout3", file: !5, line: 76, size: 128, align: 32, elements: !37)
!37 = !{!38, !39, !44}
!38 = !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs0", scope: !36, file: !5, line: 77, baseType: !9, size: 8, align: 8)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs4", scope: !36, file: !5, line: 78, baseType: !40, size: 64, align: 32, offset: 32)
!40 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8_pack4", file: !5, line: 72, size: 64, align: 32, elements: !41)
!41 = !{!42, !43}
!42 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !40, file: !5, line: 73, baseType: !14, size: 4, align: 32)
!43 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !40, file: !5, line: 74, baseType: !16, size: 60, offset: 4)
!44 = !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs12", scope: !36, file: !5, line: 79, baseType: !14, size: 1, align: 32, offset: 96)
!45 = !{i32 2, !"Dwarf Version", i32 2}
!46 = !{i32 2, !"Debug Info Version", i32 3}
!47 = !{!"clang version 3.7.0 (trunk 240791) (llvm/trunk 240790)"}
