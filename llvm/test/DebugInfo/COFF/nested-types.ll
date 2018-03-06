; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; C++ source to regenerate:
; struct HasNested {
;   enum InnerEnum { _BUF_SIZE = 1 };
;   typedef int InnerTypedef;
;   enum { InnerEnumerator = 2 };
;   struct InnerStruct { };
; };
; HasNested f;
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

%struct.HasNested = type { i8 }

@"\01?f@@3UHasNested@@A" = global %struct.HasNested zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "f", linkageName: "\01?f@@3UHasNested@@A", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !16)
!3 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "40c412b85e2b27acb30eef53983b1da4")
!4 = !{!5, !10}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "InnerEnum", scope: !6, file: !3, line: 2, baseType: !9, size: 32, elements: !14, identifier: ".?AW4InnerEnum@HasNested@@")
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasNested", file: !3, line: 1, size: 8, elements: !7, identifier: ".?AUHasNested@@")
!7 = !{!5, !8, !10, !13}
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "InnerTypedef", scope: !6, file: !3, line: 3, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !6, file: !3, line: 4, baseType: !9, size: 32, elements: !11, identifier: ".?AW4<unnamed-enum-InnerEnumerator>@HasNested@@")
!11 = !{!12}
!12 = !DIEnumerator(name: "InnerEnumerator", value: 2)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "InnerStruct", scope: !6, file: !3, line: 5, flags: DIFlagFwdDecl, identifier: ".?AUInnerStruct@HasNested@@")
!14 = !{!15}
!15 = !DIEnumerator(name: "_BUF_SIZE", value: 1)
!16 = !{!0}
!17 = !{i32 2, !"CodeView", i32 1}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 2}
!20 = !{i32 7, !"PIC Level", i32 2}
!21 = !{!"clang version 6.0.0 "}

; InnerEnum:

; CHECK: FieldList ([[INNERENUM_MEMBERS:0x.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:   Enumerator {
; CHECK-NEXT:     TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     EnumValue: 1
; CHECK-NEXT:     Name: _BUF_SIZE
; CHECK-NEXT:   }
; CHECK-NEXT: }
;
; CHECK:      Enum ([[INNERENUM:0x.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:   NumEnumerators: 1
; CHECK-NEXT:   Properties [ (0x208)
; CHECK-NEXT:     HasUniqueName (0x200)
; CHECK-NEXT:     Nested (0x8)
; CHECK-NEXT:   ]
; CHECK-NEXT:   UnderlyingType: int (0x74)
; CHECK-NEXT:   FieldListType: <field list> ([[INNERENUM_MEMBERS]])
; CHECK-NEXT:   Name: HasNested::InnerEnum
; CHECK-NEXT:   LinkageName: .?AW4InnerEnum@HasNested@@
; CHECK-NEXT: }

; CHECK:      FieldList ([[UNNAMED_MEMBERS:0x.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:   Enumerator {
; CHECK-NEXT:     TypeLeafKind: LF_ENUMERATE (0x1502)
; CHECK-NEXT:     AccessSpecifier: Public (0x3)
; CHECK-NEXT:     EnumValue: 2
; CHECK-NEXT:     Name: InnerEnumerator
; CHECK-NEXT:   }
; CHECK-NEXT: }
;
; CHECK:      Enum ([[UNNAMEDENUM:0x.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_ENUM (0x1507)
; CHECK-NEXT:   NumEnumerators: 1
; CHECK-NEXT:   Properties [ (0x208)
; CHECK-NEXT:     HasUniqueName (0x200)
; CHECK-NEXT:     Nested (0x8)
; CHECK-NEXT:   ]
; CHECK-NEXT:   UnderlyingType: int (0x74)
; CHECK-NEXT:   FieldListType: <field list> ([[UNNAMED_MEMBERS]])
; CHECK-NEXT:   Name: HasNested::<unnamed-tag>
; CHECK-NEXT:   LinkageName: .?AW4<unnamed-enum-InnerEnumerator>@HasNested@@
; CHECK-NEXT: }

; CHECK:      Struct ([[INNERSTRUCT:0x.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:   MemberCount: 0
; CHECK-NEXT:   Properties [ (0x288)
; CHECK-NEXT:     ForwardReference (0x80)
; CHECK-NEXT:     HasUniqueName (0x200)
; CHECK-NEXT:     Nested (0x8)
; CHECK-NEXT:   ]
; CHECK-NEXT:   FieldList: 0x0
; CHECK-NEXT:   DerivedFrom: 0x0
; CHECK-NEXT:   VShape: 0x0
; CHECK-NEXT:   SizeOf: 0
; CHECK-NEXT:   Name: HasNested::InnerStruct
; CHECK-NEXT:   LinkageName: .?AUInnerStruct@HasNested@@
; CHECK-NEXT: }

; CHECK:      FieldList ([[HASNESTED_MEMBERS:0x.*]]) {
; CHECK-NEXT:   TypeLeafKind: LF_FIELDLIST (0x1203)
; CHECK-NEXT:   NestedType {
; CHECK-NEXT:     TypeLeafKind: LF_NESTTYPE (0x1510)
; CHECK-NEXT:     Type: HasNested::InnerEnum ([[INNERENUM]])
; CHECK-NEXT:     Name: InnerEnum
; CHECK-NEXT:   }
; CHECK-NEXT:   NestedType {
; CHECK-NEXT:     TypeLeafKind: LF_NESTTYPE (0x1510)
; CHECK-NEXT:     Type: int (0x74)
; CHECK-NEXT:     Name: InnerTypedef
; CHECK-NEXT:   }
; CHECK-NEXT:   NestedType {
; CHECK-NEXT:     TypeLeafKind: LF_NESTTYPE (0x1510)
; CHECK-NEXT:     Type: HasNested::<unnamed-tag> ([[UNNAMEDENUM]])
; CHECK-NEXT:     Name: {{$}}
; CHECK-NEXT:   }
; CHECK-NEXT:   NestedType {
; CHECK-NEXT:     TypeLeafKind: LF_NESTTYPE (0x1510)
; CHECK-NEXT:     Type: HasNested::InnerStruct ([[INNERSTRUCT]])
; CHECK-NEXT:     Name: InnerStruct
; CHECK-NEXT:   }
; CHECK-NEXT: }
;
; CHECK:      Struct (0x{{.*}}) {
; CHECK-NEXT:   TypeLeafKind: LF_STRUCTURE (0x1505)
; CHECK-NEXT:   MemberCount: 4
; CHECK-NEXT:   Properties [ (0x210)
; CHECK-NEXT:     ContainsNestedClass (0x10)
; CHECK-NEXT:     HasUniqueName (0x200)
; CHECK-NEXT:   ]
; CHECK-NEXT:   FieldList: <field list> ([[HASNESTED_MEMBERS]])
; CHECK-NEXT:   DerivedFrom: 0x0
; CHECK-NEXT:   VShape: 0x0
; CHECK-NEXT:   SizeOf: 1
; CHECK-NEXT:   Name: HasNested
; CHECK-NEXT:   LinkageName: .?AUHasNested@@
; CHECK-NEXT: }
