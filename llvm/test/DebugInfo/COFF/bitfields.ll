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
; CHECK:  FieldList ([[anon_fl:.*]]) {
; CHECK:    TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK:    DataMember {
; CHECK:      TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:      Type: char (0x70)
; CHECK:      FieldOffset: 0x0
; CHECK:      Name: c
; CHECK:    }
; CHECK:    DataMember {
; CHECK:      TypeLeafKind: LF_MEMBER (0x150D)
; CHECK:      Type: short (0x11)
; CHECK:      FieldOffset: 0x1
; CHECK:      Name: s
; CHECK:    }
; CHECK:  }
; CHECK:  Struct ([[anon_ty:.*]]) {
; CHECK:    TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK:    MemberCount: 2
; CHECK:    Properties [ (0x8)
; CHECK:      Nested (0x8)
; CHECK:    ]
; CHECK:    FieldList: <field list> ([[anon_fl]])
; CHECK:    SizeOf: 3
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

source_filename = "test/DebugInfo/COFF/bitfields.ll"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "x86_64-pc-windows-msvc18.0.0"

%struct.S0 = type <{ i8, i16 }>
%struct.S1 = type <{ [2 x i8], i8, i32, i32, %struct.anon, i16 }>
%struct.anon = type <{ i8, i16 }>
%struct.S2 = type { i32 }

@s0 = common global %struct.S0 zeroinitializer, align 1, !dbg !0
@s1 = common global %struct.S1 zeroinitializer, align 1, !dbg !6
@s2 = common global %struct.S2 zeroinitializer, align 1, !dbg !28

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!36, !37, !38}
!llvm.ident = !{!39}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "s0", scope: !2, file: !8, line: 7, type: !33, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (trunk 273812) (llvm/trunk 273843)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "-", directory: "/usr/local/google/home/majnemer/llvm/src")
!4 = !{}
!5 = !{!0, !6, !28}
!6 = distinct !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "s1", scope: !2, file: !8, line: 18, type: !9, isLocal: false, isDefinition: true)
!8 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/majnemer/llvm/src")
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S1", file: !8, line: 10, size: 128, elements: !10)
!10 = !{!11, !16, !17, !19, !20, !21, !27}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "x1", scope: !9, file: !8, line: 11, baseType: !12, size: 16)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 16, elements: !14)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !{!15}
!15 = !DISubrange(count: 2)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "x2", scope: !9, file: !8, line: 12, baseType: !13, size: 8, offset: 16)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !9, file: !8, line: 13, baseType: !18, size: 23, offset: 24, flags: DIFlagBitField, extraData: i64 24)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !9, file: !8, line: 14, baseType: !18, size: 23, offset: 56, flags: DIFlagBitField, extraData: i64 56)
!20 = !DIDerivedType(tag: DW_TAG_member, name: "w", scope: !9, file: !8, line: 15, baseType: !18, size: 2, offset: 79, flags: DIFlagBitField, extraData: i64 56)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !9, file: !8, line: 16, baseType: !22, size: 24, offset: 88)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !9, file: !8, line: 16, size: 24, elements: !23)
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !22, file: !8, line: 16, baseType: !13, size: 8)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !22, file: !8, line: 16, baseType: !26, size: 16, offset: 8)
!26 = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !9, file: !8, line: 17, baseType: !26, size: 3, offset: 112, flags: DIFlagBitField, extraData: i64 112)
!28 = distinct !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = !DIGlobalVariable(name: "s2", scope: !2, file: !8, line: 24, type: !30, isLocal: false, isDefinition: true)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S2", file: !8, line: 21, size: 32, elements: !31)
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !30, file: !8, line: 23, baseType: !18, size: 1, flags: DIFlagBitField, extraData: i64 0)
!33 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: !8, line: 3, size: 24, elements: !34)
!34 = !{!35}
!35 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !33, file: !8, line: 6, baseType: !26, size: 8, offset: 16, flags: DIFlagBitField, extraData: i64 8)
!36 = !{i32 2, !"CodeView", i32 1}
!37 = !{i32 2, !"Debug Info Version", i32 3}
!38 = !{i32 1, !"PIC Level", i32 2}
!39 = !{!"clang version 3.9.0 (trunk 273812) (llvm/trunk 273843)"}

