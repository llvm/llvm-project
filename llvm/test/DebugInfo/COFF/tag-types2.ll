; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck --check-prefixes=ONE,BOTH %s
; RUN: llc < %s -filetype=obj -use-codeview-tagrecord2 | llvm-readobj - --codeview | FileCheck --check-prefixes=TWO,BOTH %s

; BOTH: CodeViewTypes [
; BOTH:   Section: .debug$T (5)
; ONE:   Union (0x1000) {
; TWO:   Union2 (0x1000) {
; ONE-NEXT:      TypeLeafKind: LF_UNION (0x1506)
; TWO-NEXT:      TypeLeafKind: LF_UNION2 (0x160A)
; BOTH-NEXT:     MemberCount: 0
; BOTH-NEXT:     Properties [ (0x280)
; BOTH-NEXT:       ForwardReference (0x80)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:     ]
; BOTH-NEXT:     FieldList: 0x0
; BOTH-NEXT:     SizeOf: 0
; BOTH-NEXT:     Name: AUnion
; BOTH-NEXT:     LinkageName: .?ATAUnion@@
; BOTH-NEXT:   }
; BOTH:   FieldList (0x1001) {
; BOTH:     TypeLeafKind: LF_FIELDLIST (0x1203)
; BOTH:   }
; BOTH:   Enum (0x1002) {
; BOTH-NEXT:     TypeLeafKind: LF_ENUM (0x1507)
; BOTH-NEXT:     NumEnumerators: 3
; BOTH-NEXT:     Properties [ (0x200)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:     ]
; BOTH-NEXT:     UnderlyingType: int (0x74)
; BOTH-NEXT:     FieldListType: <field list> (0x1001)
; BOTH-NEXT:     Name: AEnum
; BOTH-NEXT:     LinkageName: .?AW4AEnum@@
; BOTH-NEXT:   }
; BOTH:   StringId (0x1003) {
; BOTH:   }
; BOTH:   UdtSourceLine (0x1004) {
; BOTH:   }
; BOTH:   Pointer (0x1005) {
; BOTH:     PointeeType: AUnion (0x1000)
; BOTH:   }
; ONE:   Class (0x1006) {
; TWO:   Class2 (0x1006) {
; ONE-NEXT:      TypeLeafKind: LF_CLASS (0x1504)
; TWO-NEXT:      TypeLeafKind: LF_CLASS2 (0x1608)
; BOTH-NEXT:     MemberCount: 0
; BOTH-NEXT:     Properties [ (0x280)
; BOTH-NEXT:       ForwardReference (0x80)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:     ]
; BOTH-NEXT:     FieldList: 0x0
; BOTH-NEXT:     DerivedFrom: 0x0
; BOTH-NEXT:     VShape: 0x0
; BOTH-NEXT:     SizeOf: 0
; BOTH-NEXT:     Name: AClass
; BOTH-NEXT:     LinkageName: .?AVAClass@@
; BOTH-NEXT:   }
; ONE:   Struct (0x1007) {
; TWO:   Struct2 (0x1007) {
; ONE-NEXT:      TypeLeafKind: LF_STRUCTURE (0x1505)
; TWO-NEXT:      TypeLeafKind: LF_STRUCTURE2 (0x1609)
; BOTH-NEXT:     MemberCount: 0
; BOTH-NEXT:     Properties [ (0x280)
; BOTH-NEXT:       ForwardReference (0x80)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:     ]
; BOTH-NEXT:     FieldList: 0x0
; BOTH-NEXT:     DerivedFrom: 0x0
; BOTH-NEXT:     VShape: 0x0
; BOTH-NEXT:     SizeOf: 0
; BOTH-NEXT:     Name: AStruct
; BOTH-NEXT:     LinkageName: .?AUAStruct@@
; BOTH-NEXT:   }
; BOTH:   FieldList (0x1008) {
; BOTH:     TypeLeafKind: LF_FIELDLIST (0x1203)
; BOTH:   }
; ONE:   Union (0x1009) {
; TWO:   Union2 (0x1009) {
; ONE-NEXT:      TypeLeafKind: LF_UNION (0x1506)
; TWO-NEXT:      TypeLeafKind: LF_UNION2 (0x160A)
; BOTH-NEXT:     MemberCount: 4
; BOTH-NEXT:     Properties [ (0x600)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:       Sealed (0x400)
; BOTH-NEXT:     ]
; BOTH-NEXT:     FieldList: <field list> (0x1008)
; BOTH-NEXT:     SizeOf: 8
; BOTH-NEXT:     Name: AUnion
; BOTH-NEXT:     LinkageName: .?ATAUnion@@
; BOTH-NEXT:   }
; BOTH:   UdtSourceLine (0x100A) {
; BOTH:   }
; BOTH:   FieldList (0x100B) {
; BOTH:     TypeLeafKind: LF_FIELDLIST (0x1203)
; BOTH:   }
; ONE:   Class (0x100C) {
; TWO:   Class2 (0x100C) {
; ONE-NEXT:      TypeLeafKind: LF_CLASS (0x1504)
; TWO-NEXT:      TypeLeafKind: LF_CLASS2 (0x1608)
; BOTH-NEXT:     MemberCount: 1
; BOTH-NEXT:     Properties [ (0x200)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:     ]
; BOTH-NEXT:     FieldList: <field list> (0x100B)
; BOTH-NEXT:     DerivedFrom: 0x0
; BOTH-NEXT:     VShape: 0x0
; BOTH-NEXT:     SizeOf: 8
; BOTH-NEXT:     Name: AClass
; BOTH-NEXT:     LinkageName: .?AVAClass@@
; BOTH-NEXT:   }
; BOTH:   UdtSourceLine (0x100D) {
; BOTH:   }
; BOTH:   Pointer (0x100E) {
; BOTH:     PointeeType: AClass (0x1006)
; BOTH:   }
; BOTH:   FieldList (0x100F) {
; BOTH:     TypeLeafKind: LF_FIELDLIST (0x1203)
; BOTH:   }
; ONE:   Struct (0x1010) {
; TWO:   Struct2 (0x1010) {
; ONE-NEXT:      TypeLeafKind: LF_STRUCTURE (0x1505)
; TWO-NEXT:      TypeLeafKind: LF_STRUCTURE2 (0x1609)
; BOTH-NEXT:     MemberCount: 1
; BOTH-NEXT:     Properties [ (0x200)
; BOTH-NEXT:       HasUniqueName (0x200)
; BOTH-NEXT:     ]
; BOTH-NEXT:     FieldList: <field list> (0x100F)
; BOTH-NEXT:     DerivedFrom: 0x0
; BOTH-NEXT:     VShape: 0x0
; BOTH-NEXT:     SizeOf: 8
; BOTH-NEXT:     Name: AStruct
; BOTH-NEXT:     LinkageName: .?AUAStruct@@
; BOTH-NEXT:   }
; BOTH: ]

; BOTH: Subsection [
; BOTH:   SubSectionType: Symbols (0xF1)
; BOTH:   SubSectionSize: 0x30
; BOTH:   UDTSym {
; BOTH:     Kind: S_UDT (0x1108)
; BOTH:     Type: AUnion (0x1009)
; BOTH:     UDTName: AUnion
; BOTH:   }
; BOTH:   UDTSym {
; BOTH:     Kind: S_UDT (0x1108)
; BOTH:     Type: AClass (0x100C)
; BOTH:     UDTName: AClass
; BOTH:   }
; BOTH:   UDTSym {
; BOTH:     Kind: S_UDT (0x1108)
; BOTH:     Type: AStruct (0x1010)
; BOTH:     UDTName: AStruct
; BOTH:   }
; BOTH: ]

source_filename = "tag-types.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.51.36260"

%union.AUnion = type { ptr }

@"?u@@3TAUnion@@A" = dso_local global %union.AUnion zeroinitializer, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!26, !27, !28, !29, !30}
!llvm.ident = !{!31}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "u", linkageName: "?u@@3TAUnion@@A", scope: !2, file: !3, line: 17, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !11, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "tag-types.cpp", directory: "/tmp")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AEnum", file: !3, line: 9, baseType: !6, size: 32, elements: !7, identifier: ".?AW4AEnum@@")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8, !9, !10}
!8 = !DIEnumerator(name: "Foo", value: 0)
!9 = !DIEnumerator(name: "Bar", value: 1)
!10 = !DIEnumerator(name: "Baz", value: 2)
!11 = !{!0}
!12 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "AUnion", file: !3, line: 10, size: 64, flags: DIFlagTypePassByValue, elements: !13, identifier: ".?ATAUnion@@")
!13 = !{!14, !15, !17, !21}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !12, file: !3, line: 11, baseType: !5, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !12, file: !3, line: 12, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !12, file: !3, line: 13, baseType: !18, size: 64)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "AClass", file: !3, line: 6, size: 64, flags: DIFlagTypePassByValue, elements: !19, identifier: ".?AVAClass@@")
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "u", scope: !18, file: !3, line: 7, baseType: !16, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "s", scope: !12, file: !3, line: 14, baseType: !22, size: 64)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "AStruct", file: !3, line: 2, size: 64, flags: DIFlagTypePassByValue, elements: !23, identifier: ".?AUAStruct@@")
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "clazz", scope: !22, file: !3, line: 3, baseType: !25, size: 64)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!26 = !{i32 2, !"CodeView", i32 1}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{i32 8, !"PIC Level", i32 2}
!29 = !{i32 7, !"uwtable", i32 2}
!30 = !{i32 1, !"MaxTLSAlign", i32 65536}
!31 = !{!"clang"}
