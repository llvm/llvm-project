; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck %s

; C++ records list base classes, static data members and methods among their
; elements. A non-virtual base becomes an anonymous member of the base type at
; its offset, so the layout stays complete. Static data members and methods
; are dropped, and so is a virtual base: its offset is only known at run time,
; and the DI offset of a virtual base is the position of its offset in the
; vtable. The declaration tag index counts the emitted members. An empty base
; shares the offset of the first field and comes first, a base before
; bit-fields gets bitfield_size=0, and map definitions skip the same elements.
;
; struct Base { int b; };
; struct Base2 { int c; };
; struct S : Base, Base2 {
;   static int s1;       // DW_TAG_variable
;   int x;
;   void m();
;   static int s2;       // DW_TAG_member with DIFlagStaticMember
;   int y __attribute__((btf_decl_tag("y_tag")));
; };
; struct V : virtual Base { int v; };   // vptr at 0, v at 8, Base at 12
; struct Empty {};
; struct E : Empty { int e; };          // Empty and e both at offset 0
; struct BF : Base { int x : 3; int y : 5; };
; struct Map { int *key; void lookup(); static int instances; int *value; };
; S value;
; V vvalue;
; E evalue;
; BF bfvalue;
; Map map __attribute__((section(".maps")));
;
; The vptr member of V ('_vptr$V') is left out of the debug info below on
; purpose: its name is not valid in BTF, which is dealt with separately.

; CHECK:      [1] PTR '(anon)' type_id=2
; CHECK-NEXT: [2] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [3] STRUCT 'Map' size=16 vlen=2
; CHECK-NEXT:         'key' type_id=1 bits_offset=0
; CHECK-NEXT:         'value' type_id=1 bits_offset=64
; CHECK-NEXT: [4] VAR 'map' type_id=3, linkage=global
; CHECK-NEXT: [5] STRUCT 'S' size=16 vlen=4
; CHECK-NEXT:         '(anon)' type_id=6 bits_offset=0
; CHECK-NEXT:         '(anon)' type_id=7 bits_offset=32
; CHECK-NEXT:         'x' type_id=2 bits_offset=64
; CHECK-NEXT:         'y' type_id=2 bits_offset=96
; CHECK-NEXT: [6] STRUCT 'Base' size=4 vlen=1
; CHECK-NEXT:         'b' type_id=2 bits_offset=0
; CHECK-NEXT: [7] STRUCT 'Base2' size=4 vlen=1
; CHECK-NEXT:         'c' type_id=2 bits_offset=0
; CHECK-NEXT: [8] DECL_TAG 'y_tag' type_id=5 component_idx=3
; CHECK-NEXT: [9] VAR 'value' type_id=5, linkage=global
; CHECK-NEXT: [10] STRUCT 'V' size=16 vlen=1
; CHECK-NEXT:         'v' type_id=2 bits_offset=64
; CHECK-NEXT: [11] VAR 'vvalue' type_id=10, linkage=global
; CHECK-NEXT: [12] STRUCT 'E' size=4 vlen=2
; CHECK-NEXT:         '(anon)' type_id=13 bits_offset=0
; CHECK-NEXT:         'e' type_id=2 bits_offset=0
; CHECK-NEXT: [13] STRUCT 'Empty' size=1 vlen=0
; CHECK-NEXT: [14] VAR 'evalue' type_id=12, linkage=global
; CHECK-NEXT: [15] STRUCT 'BF' size=8 vlen=3
; CHECK-NEXT:         '(anon)' type_id=6 bits_offset=0 bitfield_size=0
; CHECK-NEXT:         'x' type_id=2 bits_offset=32 bitfield_size=3
; CHECK-NEXT:         'y' type_id=2 bits_offset=35 bitfield_size=5
; CHECK-NEXT: [16] VAR 'bfvalue' type_id=15, linkage=global

%struct.S = type { i32, i32, i32, i32 }
%struct.V = type { i32, i32, i32, i32 }

@value = global %struct.S zeroinitializer, align 4, !dbg !0
@vvalue = global %struct.V zeroinitializer, align 4, !dbg !40
@evalue = global i32 0, align 4, !dbg !100
@bfvalue = global i64 0, align 4, !dbg !110
@map = global [2 x ptr] zeroinitializer, section ".maps", align 8, !dbg !120

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!30, !31}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "value", scope: !2, file: !3, line: 12, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !40, !100, !110, !120}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 3, size: 128, elements: !6)
!6 = !{!7, !23, !11, !12, !13, !17, !18}
!7 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !5, baseType: !8, extraData: i32 0)
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base", file: !3, line: 1, size: 32, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !8, file: !3, line: 1, baseType: !20, size: 32)
!11 = !DIDerivedType(tag: DW_TAG_variable, name: "s1", scope: !5, file: !3, line: 4, baseType: !20, flags: DIFlagStaticMember)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !3, line: 5, baseType: !20, size: 32, offset: 64)
!13 = !DISubprogram(name: "m", linkageName: "_ZN1S1mEv", scope: !5, file: !3, line: 6, type: !14, scopeLine: 6, flags: DIFlagPrototyped, spFlags: 0)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "s2", scope: !5, file: !3, line: 7, baseType: !20, flags: DIFlagStaticMember)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !3, line: 8, baseType: !20, size: 32, offset: 96, annotations: !21)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !{!22}
!22 = !{!"btf_decl_tag", !"y_tag"}
!23 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !5, baseType: !24, offset: 32, extraData: i32 0)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Base2", file: !3, line: 2, size: 32, elements: !25)
!25 = !{!26}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !24, file: !3, line: 2, baseType: !20, size: 32)
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{i32 2, !"Dwarf Version", i32 5}
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression())
!41 = distinct !DIGlobalVariable(name: "vvalue", scope: !2, file: !3, line: 13, type: !42, isLocal: false, isDefinition: true)
!42 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "V", file: !3, line: 10, size: 128, elements: !43)
!43 = !{!44, !45}
!44 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !42, baseType: !8, offset: 24, flags: DIFlagVirtual, extraData: i32 0)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "v", scope: !42, file: !3, line: 10, baseType: !20, size: 32, offset: 64)
!100 = !DIGlobalVariableExpression(var: !101, expr: !DIExpression())
!101 = distinct !DIGlobalVariable(name: "evalue", scope: !2, file: !3, line: 16, type: !102, isLocal: false, isDefinition: true)
!102 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "E", file: !3, line: 15, size: 32, elements: !103)
!103 = !{!104, !106}
!104 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !102, baseType: !105, extraData: i32 0)
!105 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Empty", file: !3, line: 14, size: 8, elements: !107)
!106 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !102, file: !3, line: 15, baseType: !20, size: 32)
!107 = !{}
!110 = !DIGlobalVariableExpression(var: !111, expr: !DIExpression())
!111 = distinct !DIGlobalVariable(name: "bfvalue", scope: !2, file: !3, line: 18, type: !112, isLocal: false, isDefinition: true)
!112 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "BF", file: !3, line: 17, size: 64, elements: !113)
!113 = !{!114, !115, !116}
!114 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !112, baseType: !8, extraData: i32 0)
!115 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !112, file: !3, line: 17, baseType: !20, size: 3, offset: 32, flags: DIFlagBitField, extraData: i64 32)
!116 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !112, file: !3, line: 17, baseType: !20, size: 5, offset: 35, flags: DIFlagBitField, extraData: i64 32)
!120 = !DIGlobalVariableExpression(var: !121, expr: !DIExpression())
!121 = distinct !DIGlobalVariable(name: "map", scope: !2, file: !3, line: 20, type: !122, isLocal: false, isDefinition: true)
!122 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Map", file: !3, line: 19, size: 128, elements: !123)
!123 = !{!124, !125, !126, !127}
!124 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !122, file: !3, line: 19, baseType: !128, size: 64)
!125 = !DISubprogram(name: "lookup", scope: !122, file: !3, line: 19, type: !129, spFlags: 0)
!126 = !DIDerivedType(tag: DW_TAG_variable, name: "instances", scope: !122, file: !3, line: 19, baseType: !20, flags: DIFlagStaticMember)
!127 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !122, file: !3, line: 19, baseType: !128, size: 64, offset: 64)
!128 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!129 = !DISubroutineType(types: !130)
!130 = !{null}
