; Generated from tools/clang/test/CodeGen/debug-info-packed-struct.c
; ModuleID = 'llvm/tools/clang/test/CodeGen/debug-info-packed-struct.c'
source_filename = "test/DebugInfo/X86/debug-info-packed-struct.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"
; RUN: llc -O0 -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -v -debug-info %t.o | FileCheck %s

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
%struct.layout1 = type <{ i8, %struct.size8_anon, i8, [2 x i8] }>
%struct.size8_anon = type { i64 }
; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout0"
; CHECK:      DW_AT_byte_size {{.*}} (0x18)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l0_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l0_ofs8"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x8)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l0_ofs16"
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x1f)
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x10)

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

%struct.layout2 = type <{ i8, %struct.size8_pack1, i8 }>
%struct.size8_pack1 = type { i64 }
%struct.layout3 = type <{ i8, [3 x i8], %struct.size8_pack4, i8, [3 x i8] }>
%struct.size8_pack4 = type { i64 }

; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout1"
; CHECK:      DW_AT_byte_size {{.*}} (0x0c)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l1_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l1_ofs1"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x1)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l1_ofs9"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x17)
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x8)

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

@l0 = common global %struct.layout0 zeroinitializer, align 8, !dbg !0
@l1 = common global %struct.layout1 zeroinitializer, align 4, !dbg !6
; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout2"
; CHECK:      DW_AT_byte_size {{.*}} (0x0a)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l2_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l2_ofs1"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x1)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l2_ofs9"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x17)
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x8)

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

@l2 = common global %struct.layout2 zeroinitializer, align 1, !dbg !17
@l3 = common global %struct.layout3 zeroinitializer, align 4, !dbg !29
; CHECK:  DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "layout3"
; CHECK:      DW_AT_byte_size {{.*}} (0x10)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l3_ofs0"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x0)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l3_ofs4"
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0x4)
; CHECK:      DW_TAG_member
; CHECK:          DW_AT_name {{.*}} "l3_ofs12"
; CHECK:          DW_AT_byte_size  {{.*}} (0x04)
; CHECK:          DW_AT_bit_size   {{.*}} (0x01)
; CHECK:          DW_AT_bit_offset {{.*}} (0x1f)
; CHECK:          DW_AT_data_member_location {{.*}} (DW_OP_plus_uconst 0xc)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!49, !50}
!llvm.ident = !{!51}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "l0", scope: !2, file: !8, line: 88, type: !40, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.7.0 (trunk 240791) (llvm/trunk 240790)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "/llvm/tools/clang/test/CodeGen/<stdin>", directory: "/llvm/_build.ninja.release")
!4 = !{}
!5 = !{!0, !6, !17, !29}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "l1", scope: !2, file: !8, line: 89, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "/llvm/tools/clang/test/CodeGen/debug-info-packed-struct.c", directory: "/llvm/_build.ninja.release")
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout1", file: !8, line: 34, size: 96, elements: !10)
!10 = !{!11, !13, !15}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs0", scope: !9, file: !8, line: 35, baseType: !12, size: 8)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs1", scope: !9, file: !8, line: 36, baseType: !14, size: 64, offset: 8)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8_anon", file: !8, line: 30, size: 64, elements: !4)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "l1_ofs9", scope: !9, file: !8, line: 37, baseType: !16, size: 1, offset: 72)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = !DIGlobalVariable(name: "l2", scope: !2, file: !8, line: 90, type: !19, isLocal: false, isDefinition: true)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout2", file: !8, line: 54, size: 80, elements: !20)
!20 = !{!21, !22, !28}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs0", scope: !19, file: !8, line: 55, baseType: !12, size: 8)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs1", scope: !19, file: !8, line: 56, baseType: !23, size: 64, offset: 8)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8_pack1", file: !8, line: 50, size: 64, elements: !24)
!24 = !{!25, !26}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !23, file: !8, line: 51, baseType: !16, size: 4)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !23, file: !8, line: 52, baseType: !27, size: 60, offset: 4)
!27 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "l2_ofs9", scope: !19, file: !8, line: 57, baseType: !16, size: 1, offset: 72)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = !DIGlobalVariable(name: "l3", scope: !2, file: !8, line: 91, type: !31, isLocal: false, isDefinition: true)
!31 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout3", file: !8, line: 76, size: 128, elements: !32)
!32 = !{!33, !34, !39}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs0", scope: !31, file: !8, line: 77, baseType: !12, size: 8)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs4", scope: !31, file: !8, line: 78, baseType: !35, size: 64, offset: 32)
!35 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8_pack4", file: !8, line: 72, size: 64, elements: !36)
!36 = !{!37, !38}
!37 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !35, file: !8, line: 73, baseType: !16, size: 4)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !35, file: !8, line: 74, baseType: !27, size: 60, offset: 4)
!39 = !DIDerivedType(tag: DW_TAG_member, name: "l3_ofs12", scope: !31, file: !8, line: 79, baseType: !16, size: 1, offset: 96)
!40 = !DICompositeType(tag: DW_TAG_structure_type, name: "layout0", file: !8, line: 15, size: 192, elements: !41)
!41 = !{!42, !43, !48}
!42 = !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs0", scope: !40, file: !8, line: 16, baseType: !12, size: 8)
!43 = !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs8", scope: !40, file: !8, line: 17, baseType: !44, size: 64, offset: 64)
!44 = !DICompositeType(tag: DW_TAG_structure_type, name: "size8", file: !8, line: 11, size: 64, elements: !45)
!45 = !{!46, !47}
!46 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !44, file: !8, line: 12, baseType: !16, size: 4)
!47 = !DIDerivedType(tag: DW_TAG_member, name: "l", scope: !44, file: !8, line: 13, baseType: !27, size: 60, offset: 4)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "l0_ofs16", scope: !40, file: !8, line: 18, baseType: !16, size: 1, offset: 128)
!49 = !{i32 2, !"Dwarf Version", i32 2}
!50 = !{i32 2, !"Debug Info Version", i32 3}
!51 = !{!"clang version 3.7.0 (trunk 240791) (llvm/trunk 240790)"}

