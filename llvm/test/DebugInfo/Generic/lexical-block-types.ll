; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s
; REQUIRES: object-emission

; inline __attribute__((always_inline))
; void removed() {
;   struct A1 { int i; };
;   typedef int Int1;
;   {
;     struct I1 { Int1 j; };
;     struct C1 { typedef char Char1; Char1 c; };
;     A1 a1; a1.i++;
;     {
;       I1 i1; i1.j++;
;       C1 c1; c1.c++;
;     }
;   }
; }
;
; __attribute__((always_inline))
; void not_removed() {
;   struct A2 { int i; };
;   typedef int Int2;
;   {
;     struct I2 { Int2 j; };
;     struct C2 { typedef char Char2; Char2 c; };
;     A2 a2; a2.i++;
;     {
;       I2 i2; i2.j++;
;       C2 c2; c2.c++;
;     }
;   }
; }
;
; void foo() {
;   struct A3 { int i; };
;   typedef int Int3;
;   {
;     struct I3 { Int3 j; };
;     {
;       struct C3 { typedef char Char3; Char3 c; };
;       A3 a3; a3.i++;
;       {
;         I3 i3; i3.j++;
;         C3 c3; c3.c++;
;       }
;     }
;   }
;   removed();
;   not_removed();
; }
;
; CHECK: DW_TAG_compile_unit

; Out-of-line definition of `not_removed()` shouldn't contain any debug info for types.
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_abstract_origin	{{.*}} "_Z11not_removedv"
; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_abstract_origin	{{.*}} "a2"
; CHECK:       DW_TAG_lexical_block
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_abstract_origin	{{.*}} "i2"
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_abstract_origin	{{.*}} "c2"
; CHECK:         NULL
; CHECK:       NULL
; CHECK:     NULL

; Abstract definition of `removed()`.
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("removed")
; CHECK:     DW_AT_inline	(DW_INL_inlined)

; I1 and C1 defined in the first lexical block, typedef Char1 is a child of C1.
; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_name	("a1")
; CHECK:         DW_AT_type	{{.*}} "A1"
; CHECK:       DW_TAG_lexical_block
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_type	{{.*}} "I1"
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_type	{{.*}} "C1"
; CHECK:         NULL
; CHECK:       DW_TAG_structure_type
; CHECK:         DW_AT_name	("I1")
; CHECK:         DW_TAG_member
; CHECK:           DW_AT_type	{{.*}} "Int1"
; CHECK:         NULL
; CHECK:       DW_TAG_structure_type
; CHECK:         DW_AT_name	("C1")
; CHECK:         DW_TAG_member
; CHECK:           DW_AT_type	{{.*}} "Char1"
; CHECK:         DW_TAG_typedef
; CHECK:           DW_AT_name	("Char1")
; CHECK:         NULL
; CHECK:       NULL

; A1 and typedef Int1 defined in the subprogram scope.
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name	("A1")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     DW_TAG_typedef
; CHECK:       DW_AT_name	("Int1")
; CHECK:     NULL

; CHECK:   DW_TAG_base_type
; CHECK:   DW_TAG_base_type

; Abstract definition of `not_removed()`.
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("not_removed")
; CHECK:     DW_AT_inline	(DW_INL_inlined)

; I2 and C2 defined in the first lexical block, typedef Char2 is a child of C2.
; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_name	("a2")
; CHECK:         DW_AT_type	{{.*}} "A2"
; CHECK:       DW_TAG_lexical_block
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_name	("i2")
; CHECK:           DW_AT_type	{{.*}} "I2"
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_name	("c2")
; CHECK:           DW_AT_type	{{.*}} "C2"
; CHECK:         NULL
; CHECK:       DW_TAG_structure_type
; CHECK:         DW_AT_name	("I2")
; CHECK:         DW_TAG_member
; CHECK:           DW_AT_type	{{.*}} "Int2"
; CHECK:         NULL
; CHECK:       DW_TAG_structure_type
; CHECK:         DW_AT_name	("C2")
; CHECK:         DW_TAG_member
; CHECK:           DW_AT_type	{{.*}} "Char2"
; CHECK:         DW_TAG_typedef
; CHECK:           DW_AT_name	("Char2")
; CHECK:         NULL
; CHECK:       NULL

; A2 and typedef Int2 defined in subprogram scope.
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name	("A2")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     DW_TAG_typedef
; CHECK:       DW_AT_name	("Int2")
; CHECK:     NULL

; Definition of `foo()`.
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("foo")

; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_lexical_block
; CHECK:         DW_TAG_variable
; CHECK:           DW_AT_name	("a3")
; CHECK:           DW_AT_type	{{.*}} "A3"
; CHECK:         DW_TAG_lexical_block
; CHECK:           DW_TAG_variable
; CHECK:             DW_AT_name	("i3")
; CHECK:             DW_AT_type	{{.*}} "I3"
; CHECK:           DW_TAG_variable
; CHECK:             DW_AT_name	("c3")
; CHECK:             DW_AT_type	{{.*}} "C3"
; CHECK:           NULL

; C3 has the inner lexical block scope, typedef Char3 is a child of C3.
; CHECK:         DW_TAG_structure_type
; CHECK:           DW_AT_name	("C3")
; CHECK:           DW_TAG_member
; CHECK:             DW_AT_type	{{.*}} "Char3"
; CHECK:           DW_TAG_typedef
; CHECK:             DW_AT_name	("Char3")
; CHECK:           NULL
; CHECK:         NULL

; I3 has the outer lexical block scope.
; CHECK:       DW_TAG_structure_type
; CHECK:         DW_AT_name	("I3")
; CHECK:         DW_TAG_member
; CHECK:           DW_AT_type	{{.*}} "Int3"
; CHECK:         NULL
; CHECK:       NULL

; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_AT_abstract_origin	{{.*}} "_Z7removedv"
; CHECK:       DW_TAG_lexical_block
; CHECK:         DW_TAG_variable
; CHECK:         DW_TAG_lexical_block
; CHECK:           DW_TAG_variable
; CHECK:           DW_TAG_variable
; CHECK:           NULL
; CHECK:         NULL
; CHECK:       NULL

; CHECK:     DW_TAG_inlined_subroutine
; CHECK:       DW_AT_abstract_origin	{{.*}} "_Z11not_removedv"
; CHECK:       DW_TAG_lexical_block
; CHECK:         DW_TAG_variable
; CHECK:         DW_TAG_lexical_block
; CHECK:           DW_TAG_variable
; CHECK:           DW_TAG_variable
; CHECK:           NULL
; CHECK:         NULL
; CHECK:       NULL

; A3 and Int3 defined within the subprogam scope.
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name	("A3")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     DW_TAG_typedef
; CHECK:       DW_AT_name	("Int3")
; CHECK:     NULL
; CHECK:   NULL

%struct.A2 = type { i32 }
%struct.I2 = type { i32 }
%struct.C2 = type { i8 }
%struct.A1 = type { i32 }
%struct.I1 = type { i32 }
%struct.C1 = type { i8 }
%struct.A3 = type { i32 }
%struct.I3 = type { i32 }
%struct.C3 = type { i8 }

define dso_local void @_Z11not_removedv() !dbg !8 {
entry:
  %a2 = alloca %struct.A2, align 4
  %i2 = alloca %struct.I2, align 4
  %c2 = alloca %struct.C2, align 1
  call void @llvm.dbg.declare(metadata %struct.A2* %a2, metadata !12, metadata !DIExpression()), !dbg !18
  %i = getelementptr inbounds %struct.A2, %struct.A2* %a2, i32 0, i32 0, !dbg !19
  %0 = load i32, i32* %i, align 4, !dbg !20
  %inc = add nsw i32 %0, 1, !dbg !20
  store i32 %inc, i32* %i, align 4, !dbg !20
  call void @llvm.dbg.declare(metadata %struct.I2* %i2, metadata !21, metadata !DIExpression()), !dbg !27
  %j = getelementptr inbounds %struct.I2, %struct.I2* %i2, i32 0, i32 0, !dbg !28
  %1 = load i32, i32* %j, align 4, !dbg !29
  %inc1 = add nsw i32 %1, 1, !dbg !29
  store i32 %inc1, i32* %j, align 4, !dbg !29
  call void @llvm.dbg.declare(metadata %struct.C2* %c2, metadata !30, metadata !DIExpression()), !dbg !36
  %c = getelementptr inbounds %struct.C2, %struct.C2* %c2, i32 0, i32 0, !dbg !37
  %2 = load i8, i8* %c, align 1, !dbg !38
  %inc2 = add i8 %2, 1, !dbg !38
  store i8 %inc2, i8* %c, align 1, !dbg !38
  ret void, !dbg !39
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define dso_local void @_Z3foov() !dbg !40 {
entry:
  %a1.i = alloca %struct.A1, align 4
  %i1.i = alloca %struct.I1, align 4
  %c1.i = alloca %struct.C1, align 1
  %a2.i = alloca %struct.A2, align 4
  %i2.i = alloca %struct.I2, align 4
  %c2.i = alloca %struct.C2, align 1
  %a3 = alloca %struct.A3, align 4
  %i3 = alloca %struct.I3, align 4
  %c3 = alloca %struct.C3, align 1
  call void @llvm.dbg.declare(metadata %struct.A3* %a3, metadata !41, metadata !DIExpression()), !dbg !47
  %i = getelementptr inbounds %struct.A3, %struct.A3* %a3, i32 0, i32 0, !dbg !48
  %0 = load i32, i32* %i, align 4, !dbg !49
  %inc = add nsw i32 %0, 1, !dbg !49
  store i32 %inc, i32* %i, align 4, !dbg !49
  call void @llvm.dbg.declare(metadata %struct.I3* %i3, metadata !50, metadata !DIExpression()), !dbg !56
  %j = getelementptr inbounds %struct.I3, %struct.I3* %i3, i32 0, i32 0, !dbg !57
  %1 = load i32, i32* %j, align 4, !dbg !58
  %inc1 = add nsw i32 %1, 1, !dbg !58
  store i32 %inc1, i32* %j, align 4, !dbg !58
  call void @llvm.dbg.declare(metadata %struct.C3* %c3, metadata !59, metadata !DIExpression()), !dbg !64
  %c = getelementptr inbounds %struct.C3, %struct.C3* %c3, i32 0, i32 0, !dbg !65
  %2 = load i8, i8* %c, align 1, !dbg !66
  %inc2 = add i8 %2, 1, !dbg !66
  store i8 %inc2, i8* %c, align 1, !dbg !66
  call void @llvm.dbg.declare(metadata %struct.A1* %a1.i, metadata !67, metadata !DIExpression()), !dbg !73
  %i.i3 = getelementptr inbounds %struct.A1, %struct.A1* %a1.i, i32 0, i32 0, !dbg !75
  %3 = load i32, i32* %i.i3, align 4, !dbg !76
  %inc.i4 = add nsw i32 %3, 1, !dbg !76
  store i32 %inc.i4, i32* %i.i3, align 4, !dbg !76
  call void @llvm.dbg.declare(metadata %struct.I1* %i1.i, metadata !77, metadata !DIExpression()), !dbg !83
  %j.i5 = getelementptr inbounds %struct.I1, %struct.I1* %i1.i, i32 0, i32 0, !dbg !84
  %4 = load i32, i32* %j.i5, align 4, !dbg !85
  %inc1.i6 = add nsw i32 %4, 1, !dbg !85
  store i32 %inc1.i6, i32* %j.i5, align 4, !dbg !85
  call void @llvm.dbg.declare(metadata %struct.C1* %c1.i, metadata !86, metadata !DIExpression()), !dbg !91
  %c.i7 = getelementptr inbounds %struct.C1, %struct.C1* %c1.i, i32 0, i32 0, !dbg !92
  %5 = load i8, i8* %c.i7, align 1, !dbg !93
  %inc2.i8 = add i8 %5, 1, !dbg !93
  store i8 %inc2.i8, i8* %c.i7, align 1, !dbg !93
  call void @llvm.dbg.declare(metadata %struct.A2* %a2.i, metadata !12, metadata !DIExpression()), !dbg !94
  %i.i = getelementptr inbounds %struct.A2, %struct.A2* %a2.i, i32 0, i32 0, !dbg !96
  %6 = load i32, i32* %i.i, align 4, !dbg !97
  %inc.i = add nsw i32 %6, 1, !dbg !97
  store i32 %inc.i, i32* %i.i, align 4, !dbg !97
  call void @llvm.dbg.declare(metadata %struct.I2* %i2.i, metadata !21, metadata !DIExpression()), !dbg !98
  %j.i = getelementptr inbounds %struct.I2, %struct.I2* %i2.i, i32 0, i32 0, !dbg !99
  %7 = load i32, i32* %j.i, align 4, !dbg !100
  %inc1.i = add nsw i32 %7, 1, !dbg !100
  store i32 %inc1.i, i32* %j.i, align 4, !dbg !100
  call void @llvm.dbg.declare(metadata %struct.C2* %c2.i, metadata !30, metadata !DIExpression()), !dbg !101
  %c.i = getelementptr inbounds %struct.C2, %struct.C2* %c2.i, i32 0, i32 0, !dbg !102
  %8 = load i8, i8* %c.i, align 1, !dbg !103
  %inc2.i = add i8 %8, 1, !dbg !103
  store i8 %inc2.i, i8* %c.i, align 1, !dbg !103
  ret void, !dbg !104
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "not_removed", linkageName: "_Z11not_removedv", scope: !1, file: !1, line: 17, type: !9, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !105)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{}
!12 = !DILocalVariable(name: "a2", scope: !13, file: !1, line: 23, type: !14)
!13 = distinct !DILexicalBlock(scope: !8, file: !1, line: 20, column: 3)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A2", scope: !8, file: !1, line: 18, size: 32, flags: DIFlagTypePassByValue, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !14, file: !1, line: 18, baseType: !17, size: 32)
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DILocation(line: 23, column: 8, scope: !13)
!19 = !DILocation(line: 23, column: 15, scope: !13)
!20 = !DILocation(line: 23, column: 16, scope: !13)
!21 = !DILocalVariable(name: "i2", scope: !22, file: !1, line: 25, type: !23)
!22 = distinct !DILexicalBlock(scope: !13, file: !1, line: 24, column: 5)
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "I2", scope: !13, file: !1, line: 21, size: 32, flags: DIFlagTypePassByValue, elements: !24)
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !23, file: !1, line: 21, baseType: !26, size: 32)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "Int2", scope: !8, file: !1, line: 19, baseType: !17)
!27 = !DILocation(line: 25, column: 10, scope: !22)
!28 = !DILocation(line: 25, column: 17, scope: !22)
!29 = !DILocation(line: 25, column: 18, scope: !22)
!30 = !DILocalVariable(name: "c2", scope: !22, file: !1, line: 26, type: !31)
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C2", scope: !13, file: !1, line: 22, size: 8, flags: DIFlagTypePassByValue, elements: !32)
!32 = !{!33}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !31, file: !1, line: 22, baseType: !34, size: 8)
!34 = !DIDerivedType(tag: DW_TAG_typedef, name: "Char2", scope: !31, file: !1, line: 22, baseType: !35)
!35 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!36 = !DILocation(line: 26, column: 10, scope: !22)
!37 = !DILocation(line: 26, column: 17, scope: !22)
!38 = !DILocation(line: 26, column: 18, scope: !22)
!39 = !DILocation(line: 29, column: 1, scope: !8)
!40 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 31, type: !9, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !107)
!41 = !DILocalVariable(name: "a3", scope: !42, file: !1, line: 38, type: !44)
!42 = distinct !DILexicalBlock(scope: !43, file: !1, line: 36, column: 5)
!43 = distinct !DILexicalBlock(scope: !40, file: !1, line: 34, column: 3)
!44 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A3", scope: !40, file: !1, line: 32, size: 32, flags: DIFlagTypePassByValue, elements: !45)
!45 = !{!46}
!46 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !44, file: !1, line: 32, baseType: !17, size: 32)
!47 = !DILocation(line: 38, column: 10, scope: !42)
!48 = !DILocation(line: 38, column: 17, scope: !42)
!49 = !DILocation(line: 38, column: 18, scope: !42)
!50 = !DILocalVariable(name: "i3", scope: !51, file: !1, line: 40, type: !52)
!51 = distinct !DILexicalBlock(scope: !42, file: !1, line: 39, column: 7)
!52 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "I3", scope: !43, file: !1, line: 35, size: 32, flags: DIFlagTypePassByValue, elements: !53)
!53 = !{!54}
!54 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !52, file: !1, line: 35, baseType: !55, size: 32)
!55 = !DIDerivedType(tag: DW_TAG_typedef, name: "Int3", scope: !40, file: !1, line: 33, baseType: !17)
!56 = !DILocation(line: 40, column: 12, scope: !51)
!57 = !DILocation(line: 40, column: 19, scope: !51)
!58 = !DILocation(line: 40, column: 20, scope: !51)
!59 = !DILocalVariable(name: "c3", scope: !51, file: !1, line: 41, type: !60)
!60 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C3", scope: !42, file: !1, line: 37, size: 8, flags: DIFlagTypePassByValue, elements: !61)
!61 = !{!62}
!62 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !60, file: !1, line: 37, baseType: !63, size: 8)
!63 = !DIDerivedType(tag: DW_TAG_typedef, name: "Char3", scope: !60, file: !1, line: 37, baseType: !35)
!64 = !DILocation(line: 41, column: 12, scope: !51)
!65 = !DILocation(line: 41, column: 19, scope: !51)
!66 = !DILocation(line: 41, column: 20, scope: !51)
!67 = !DILocalVariable(name: "a1", scope: !68, file: !1, line: 8, type: !70)
!68 = distinct !DILexicalBlock(scope: !69, file: !1, line: 5, column: 3)
!69 = distinct !DISubprogram(name: "removed", linkageName: "_Z7removedv", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !110)
!70 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A1", scope: !69, file: !1, line: 3, size: 32, flags: DIFlagTypePassByValue, elements: !71, identifier: "_ZTSZ7removedvE2A1")
!71 = !{!72}
!72 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !70, file: !1, line: 3, baseType: !17, size: 32)
!73 = !DILocation(line: 8, column: 8, scope: !68, inlinedAt: !74)
!74 = distinct !DILocation(line: 45, column: 3, scope: !40)
!75 = !DILocation(line: 8, column: 15, scope: !68, inlinedAt: !74)
!76 = !DILocation(line: 8, column: 16, scope: !68, inlinedAt: !74)
!77 = !DILocalVariable(name: "i1", scope: !78, file: !1, line: 10, type: !79)
!78 = distinct !DILexicalBlock(scope: !68, file: !1, line: 9, column: 5)
!79 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "I1", scope: !68, file: !1, line: 6, size: 32, flags: DIFlagTypePassByValue, elements: !80, identifier: "_ZTSZ7removedvE2I1")
!80 = !{!81}
!81 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !79, file: !1, line: 6, baseType: !82, size: 32)
!82 = !DIDerivedType(tag: DW_TAG_typedef, name: "Int1", scope: !69, file: !1, line: 4, baseType: !17)
!83 = !DILocation(line: 10, column: 10, scope: !78, inlinedAt: !74)
!84 = !DILocation(line: 10, column: 17, scope: !78, inlinedAt: !74)
!85 = !DILocation(line: 10, column: 18, scope: !78, inlinedAt: !74)
!86 = !DILocalVariable(name: "c1", scope: !78, file: !1, line: 11, type: !87)
!87 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C1", scope: !68, file: !1, line: 7, size: 8, flags: DIFlagTypePassByValue, elements: !88, identifier: "_ZTSZ7removedvE2C1")
!88 = !{!89}
!89 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !87, file: !1, line: 7, baseType: !90, size: 8)
!90 = !DIDerivedType(tag: DW_TAG_typedef, name: "Char1", scope: !87, file: !1, line: 7, baseType: !35)
!91 = !DILocation(line: 11, column: 10, scope: !78, inlinedAt: !74)
!92 = !DILocation(line: 11, column: 17, scope: !78, inlinedAt: !74)
!93 = !DILocation(line: 11, column: 18, scope: !78, inlinedAt: !74)
!94 = !DILocation(line: 23, column: 8, scope: !13, inlinedAt: !95)
!95 = distinct !DILocation(line: 46, column: 3, scope: !40)
!96 = !DILocation(line: 23, column: 15, scope: !13, inlinedAt: !95)
!97 = !DILocation(line: 23, column: 16, scope: !13, inlinedAt: !95)
!98 = !DILocation(line: 25, column: 10, scope: !22, inlinedAt: !95)
!99 = !DILocation(line: 25, column: 17, scope: !22, inlinedAt: !95)
!100 = !DILocation(line: 25, column: 18, scope: !22, inlinedAt: !95)
!101 = !DILocation(line: 26, column: 10, scope: !22, inlinedAt: !95)
!102 = !DILocation(line: 26, column: 17, scope: !22, inlinedAt: !95)
!103 = !DILocation(line: 26, column: 18, scope: !22, inlinedAt: !95)
!104 = !DILocation(line: 47, column: 1, scope: !40)
!105 = !{!14, !23, !26, !31}
!107 = !{!44, !52, !55, !60}
!110 = !{!70, !79, !82, !87}
