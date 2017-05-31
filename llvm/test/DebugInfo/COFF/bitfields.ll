; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; $ cat t.cpp
; #pragma pack(1)
; struct S0 {
;   char : 8;
;   short   : 8;
;   short x : 8;
; } s0;
;
; #pragma pack(1)
; struct S1 {
;   char x1[2];
;   char x2;
;   int y : 23;
;   int z : 23;
;   int w : 2;
;   struct { char c; short s; } v;
;   short u : 3;
; } s1;
;
; #pragma pack(1)
; struct S2 {
;   char : 0;
;   int y : 1;
; } s2;
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewTypes [
; CHECK:  BitField ([[S0_x:.*]]) {
; CHECK:    TypeLeafKind: LF_BITFIELD (0x1205)
; CHECK:    Type: short (0x11)
; CHECK:    BitSize: 8
; CHECK:    BitOffset: 8
; CHECK:  }
; CHECK:  FieldList ([[S0_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      Type: [[S0_x:.*]]
; CHECK:      FieldOffset: 0x1
; CHECK:      Name: x
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 1
; CHECK:    Properties [ (0x0)
; CHECK:    ]
; CHECK:    FieldList: <field list> ([[S0_fl]])
; CHECK:    SizeOf: 3
; CHECK:    Name: S0
; CHECK:  }
; CHECK:  BitField ([[S1_y_z:.*]]) {
; CHECK:    TypeLeafKind: LF_BITFIELD (0x1205)
; CHECK:    Type: int (0x74)
; CHECK:    BitSize: 23
; CHECK:    BitOffset: 0
; CHECK:  }
; CHECK:  BitField ([[S1_w:.*]]) {
; CHECK:    TypeLeafKind: LF_BITFIELD (0x1205)
; CHECK:    Type: int (0x74)
; CHECK:    BitSize: 2
; CHECK:    BitOffset: 23
; CHECK:  }
; CHECK:  Struct ([[anon_ty:.*]]) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 0
; CHECK:    Properties [ (0x88)
; CHECK:      ForwardReference (0x80)
; CHECK:      Nested (0x8)
; CHECK:    ]
; CHECK:    FieldList: 0x0
; CHECK:    SizeOf: 0
; CHECK:    Name: S1::<unnamed-tag>
; CHECK:  }
; CHECK:  BitField ([[S1_u:.*]]) {
; CHECK:    TypeLeafKind: LF_BITFIELD (0x1205)
; CHECK:    Type: short (0x11)
; CHECK:    BitSize: 3
; CHECK:    BitOffset: 0
; CHECK:  }
; CHECK:  FieldList ([[S1_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      FieldOffset: 0x0
; CHECK:      Name: x1
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: char (0x70)
; CHECK:      FieldOffset: 0x2
; CHECK:      Name: x2
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: [[S1_y_z]]
; CHECK:      FieldOffset: 0x3
; CHECK:      Name: y
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: [[S1_y_z]]
; CHECK:      FieldOffset: 0x7
; CHECK:      Name: z
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: [[S1_w]]
; CHECK:      FieldOffset: 0x7
; CHECK:      Name: w
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: S1::<unnamed-tag> ([[anon_ty]])
; CHECK:      FieldOffset: 0xB
; CHECK:      Name: v
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: [[S1_u]]
; CHECK:      FieldOffset: 0xE
; CHECK:      Name: u
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 7
; CHECK:    Properties [ (0x0)
; CHECK:    ]
; CHECK:    FieldList: <field list> ([[S1_fl]])
; CHECK:    SizeOf: 16
; CHECK:    Name: S1
; CHECK:  }
; CHECK:  FieldList ([[anon_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      Type: char (0x70)
; CHECK:      FieldOffset: 0x0
; CHECK:      Name: c
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      Type: short (0x11)
; CHECK:      FieldOffset: 0x1
; CHECK:      Name: s
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 2
; CHECK:    Properties [ (0x8)
; CHECK:      Nested (0x8)
; CHECK:    ]
; CHECK:    FieldList: <field list> ([[anon_fl]])
; CHECK:    SizeOf: 3
; CHECK:    Name: S1::<unnamed-tag>
; CHECK:  }
; CHECK:  BitField ([[S2_y:.*]]) {
; CHECK:    TypeLeafKind: LF_BITFIELD (0x1205)
; CHECK:    Type: int (0x74)
; CHECK:    BitSize: 1
; CHECK:    BitOffset: 0
; CHECK:  }
; CHECK:  FieldList ([[S2_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      Type: [[S2_y]]
; CHECK:      FieldOffset: 0x0
; CHECK:      Name: y
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ({{.*}}) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 1
; CHECK:    Properties [ (0x0)
; CHECK:    ]
; CHECK:    FieldList: <field list> ([[S2_fl]])
; CHECK:    SizeOf: 4
; CHECK:    Name: S2
; CHECK:  }

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "x86_64-pc-windows-msvc18.0.0"

%struct.S0 = type <{ i8, i16 }>
%struct.S1 = type <{ [2 x i8], i8, i32, i32, %struct.anon, i16 }>
%struct.anon = type <{ i8, i16 }>
%struct.S2 = type { i32 }

@s0 = common global %struct.S0 zeroinitializer, align 1
@s1 = common global %struct.S1 zeroinitializer, align 1
@s2 = common global %struct.S2 zeroinitializer, align 1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33, !34, !35}
!llvm.ident = !{!36}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 273812) (llvm/trunk 273843)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "-", directory: "/usr/local/google/home/majnemer/llvm/src")
!2 = !{}
!3 = !{!4, !10, !29}
!4 = distinct !DIGlobalVariable(name: "s0", scope: !0, file: !5, line: 7, type: !6, isLocal: false, isDefinition: true, variable: %struct.S0* @s0)
!5 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/majnemer/llvm/src")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: !5, line: 3, size: 24, align: 8, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !6, file: !5, line: 6, baseType: !9, size: 8, align: 16, offset: 16, flags: DIFlagBitField, extraData: i64 8)
!9 = !DIBasicType(name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!10 = distinct !DIGlobalVariable(name: "s1", scope: !0, file: !5, line: 18, type: !11, isLocal: false, isDefinition: true, variable: %struct.S1* @s1)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S1", file: !5, line: 10, size: 128, align: 8, elements: !12)
!12 = !{!13, !18, !19, !21, !22, !23, !28}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "x1", scope: !11, file: !5, line: 11, baseType: !14, size: 16, align: 8)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 16, align: 8, elements: !16)
!15 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!16 = !{!17}
!17 = !DISubrange(count: 2)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x2", scope: !11, file: !5, line: 12, baseType: !15, size: 8, align: 8, offset: 16)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !11, file: !5, line: 13, baseType: !20, size: 23, align: 32, offset: 24, flags: DIFlagBitField, extraData: i64 24)
!20 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !11, file: !5, line: 14, baseType: !20, size: 23, align: 32, offset: 56, flags: DIFlagBitField, extraData: i64 56)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "w", scope: !11, file: !5, line: 15, baseType: !20, size: 2, align: 32, offset: 79, flags: DIFlagBitField, extraData: i64 56)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !11, file: !5, line: 16, baseType: !24, size: 24, align: 8, offset: 88)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !11, file: !5, line: 16, size: 24, align: 8, elements: !25)
!25 = !{!26, !27}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !24, file: !5, line: 16, baseType: !15, size: 8, align: 8)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !24, file: !5, line: 16, baseType: !9, size: 16, align: 16, offset: 8)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !11, file: !5, line: 17, baseType: !9, size: 3, align: 16, offset: 112, flags: DIFlagBitField, extraData: i64 112)
!29 = distinct !DIGlobalVariable(name: "s2", scope: !0, file: !5, line: 24, type: !30, isLocal: false, isDefinition: true, variable: %struct.S2* @s2)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S2", file: !5, line: 21, size: 32, align: 8, elements: !31)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !30, file: !5, line: 23, baseType: !20, size: 1, align: 32, flags: DIFlagBitField, extraData: i64 0)
!33 = !{i32 2, !"CodeView", i32 1}
!34 = !{i32 2, !"Debug Info Version", i32 3}
!35 = !{i32 1, !"PIC Level", i32 2}
!36 = !{!"clang version 3.9.0 (trunk 273812) (llvm/trunk 273843)"}
