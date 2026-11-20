; RUN: %llc_dwarf -mtriple=x86_64-linux -O0 -filetype=obj < %s              \
; RUN:  | llvm-dwarfdump --show-children --name=foo - \
; RUN:  | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; foo() is emitted out-of-line and is also inlined into main(). The DIEs of the
; function-local types B and C are created when the constructors of A<B> and
; A<C> are emitted, before main() is processed, i.e. before it is known that
; foo() needs an abstract DIE. Check that both types are placed into the
; abstract tree of foo(), and that C is placed into the abstract lexical block.

; Compiled from the source below. The scope of C is changed to the lexical
; block, in which C is declared.
;
; template <typename T> struct A {
;   A(T &in) : a(in) {}
;   T a;
; };
;
; __attribute__((always_inline)) void foo() {
;   struct B { int i; };
;   B objB;
;   A<B> objA(objB);
;   {
;     struct C { int j; };
;     C objC;
;     A<C> objA2(objC);
;   }
; }
;
; int main() { foo(); }

; Concrete out-of-line tree of foo(), with no local types.
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin {{.*}} "_Z3foov"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objB"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objA"
; CHECK:   DW_TAG_lexical_block
; CHECK:     DW_AT_abstract_origin ([[ABS_LB:0x[0-9a-f]+]])
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_abstract_origin {{.*}} "objC"
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_abstract_origin {{.*}} "objA2"
; CHECK:     NULL
; CHECK:   NULL

; Abstract tree of foo(), with B and C.
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("foo")
; CHECK:   DW_AT_inline (DW_INL_inlined)
; CHECK:   DW_TAG_structure_type
; CHECK:     DW_AT_name ("B")
; CHECK:     DW_TAG_member
; CHECK:     NULL
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_name ("objB")
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_name ("objA")
; CHECK: [[ABS_LB]]: DW_TAG_lexical_block
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name ("C")
; CHECK:       DW_TAG_member
; CHECK:       NULL
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_name ("objC")
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_name ("objA2")
; CHECK:     NULL
; CHECK:   NULL

; Inlined instance of foo() in main().
; CHECK: DW_TAG_inlined_subroutine
; CHECK:   DW_AT_abstract_origin {{.*}} "_Z3foov"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objB"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objA"
; CHECK:   DW_TAG_lexical_block
; CHECK:     DW_AT_abstract_origin ([[ABS_LB]])
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_abstract_origin {{.*}} "objC"
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_abstract_origin {{.*}} "objA2"
; CHECK:     NULL
; CHECK:   NULL

%struct.B = type { i32 }
%struct.A = type { %struct.B }
%struct.C = type { i32 }
%struct.A.0 = type { %struct.C }

define dso_local void @_Z3foov() !dbg !7 {
entry:
  %objB = alloca %struct.B, align 4
  %objA = alloca %struct.A, align 4
  %objC = alloca %struct.C, align 4
  %objA2 = alloca %struct.A.0, align 4
    #dbg_declare(ptr %objB, !42, !DIExpression(), !43)
    #dbg_declare(ptr %objA, !44, !DIExpression(), !45)
  call void @_ZN1AIZ3foovE1BEC2ERS0_(ptr %objA, ptr %objB), !dbg !45
    #dbg_declare(ptr %objC, !46, !DIExpression(), !48)
    #dbg_declare(ptr %objA2, !49, !DIExpression(), !50)
  call void @_ZN1AIZ3foovE1CEC2ERS0_(ptr %objA2, ptr %objC), !dbg !50
  ret void, !dbg !51
}

define internal void @_ZN1AIZ3foovE1BEC2ERS0_(ptr %this, ptr %in) unnamed_addr align 2 !dbg !52 {
entry:
  %this.addr = alloca ptr, align 8
  %in.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !53, !DIExpression(), !55)
  store ptr %in, ptr %in.addr, align 8
    #dbg_declare(ptr %in.addr, !56, !DIExpression(), !57)
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = load ptr, ptr %in.addr, align 8, !dbg !59
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %this1, ptr align 4 %0, i64 4, i1 false), !dbg !58
  ret void, !dbg !61
}

define internal void @_ZN1AIZ3foovE1CEC2ERS0_(ptr %this, ptr %in) unnamed_addr align 2 !dbg !62 {
entry:
  %this.addr = alloca ptr, align 8
  %in.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !63, !DIExpression(), !65)
  store ptr %in, ptr %in.addr, align 8
    #dbg_declare(ptr %in.addr, !66, !DIExpression(), !67)
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = load ptr, ptr %in.addr, align 8, !dbg !69
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %this1, ptr align 4 %0, i64 4, i1 false), !dbg !68
  ret void, !dbg !70
}

define dso_local i32 @main() !dbg !71 {
entry:
  %objB.i = alloca %struct.B, align 4
  %objA.i = alloca %struct.A, align 4
  %objC.i = alloca %struct.C, align 4
  %objA2.i = alloca %struct.A.0, align 4
    #dbg_declare(ptr %objB.i, !42, !DIExpression(), !74)
    #dbg_declare(ptr %objA.i, !44, !DIExpression(), !76)
  call void @_ZN1AIZ3foovE1BEC2ERS0_(ptr %objA.i, ptr %objB.i), !dbg !76
    #dbg_declare(ptr %objC.i, !46, !DIExpression(), !77)
    #dbg_declare(ptr %objA2.i, !49, !DIExpression(), !78)
  call void @_ZN1AIZ3foovE1CEC2ERS0_(ptr %objA2.i, ptr %objC.i), !dbg !78
  ret i32 0, !dbg !79
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!34, !35}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{!3, !21}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<B>", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !4, templateParams: !19)
!4 = !{!5, !14}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, line: 3, baseType: !6, size: 32)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", scope: !7, file: !1, line: 7, size: 32, flags: DIFlagTypePassByValue, elements: !11)
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !6, file: !1, line: 7, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "A", scope: !3, file: !1, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !18}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !6, size: 64)
!19 = !{!20}
!20 = !DITemplateTypeParameter(name: "T", type: !6)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<C>", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !22, templateParams: !32)
!22 = !{!23, !27}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !21, file: !1, line: 3, baseType: !24, size: 32)
!24 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", scope: !47, file: !1, line: 11, size: 32, flags: DIFlagTypePassByValue, elements: !25)
!25 = !{!26}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "j", scope: !24, file: !1, line: 11, baseType: !13, size: 32)
!27 = !DISubprogram(name: "A", scope: !21, file: !1, line: 2, type: !28, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30, !31}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !24, size: 64)
!32 = !{!33}
!33 = !DITemplateTypeParameter(name: "T", type: !24)
!34 = !{i32 7, !"Dwarf Version", i32 5}
!35 = !{i32 2, !"Debug Info Version", i32 3}
!42 = !DILocalVariable(name: "objB", scope: !7, file: !1, line: 8, type: !6)
!43 = !DILocation(line: 8, column: 5, scope: !7)
!44 = !DILocalVariable(name: "objA", scope: !7, file: !1, line: 9, type: !3)
!45 = !DILocation(line: 9, column: 8, scope: !7)
!46 = !DILocalVariable(name: "objC", scope: !47, file: !1, line: 12, type: !24)
!47 = distinct !DILexicalBlock(scope: !7, file: !1, line: 10, column: 3)
!48 = !DILocation(line: 12, column: 7, scope: !47)
!49 = !DILocalVariable(name: "objA2", scope: !47, file: !1, line: 13, type: !21)
!50 = !DILocation(line: 13, column: 10, scope: !47)
!51 = !DILocation(line: 15, column: 1, scope: !7)
!52 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AIZ3foovE1BEC2ERS0_", scope: !3, file: !1, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, declaration: !14, retainedNodes: !10)
!53 = !DILocalVariable(name: "this", arg: 1, scope: !52, type: !54, flags: DIFlagArtificial | DIFlagObjectPointer)
!54 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!55 = !DILocation(line: 0, scope: !52)
!56 = !DILocalVariable(name: "in", arg: 2, scope: !52, file: !1, line: 2, type: !18)
!57 = !DILocation(line: 2, column: 8, scope: !52)
!58 = !DILocation(line: 2, column: 14, scope: !52)
!59 = !DILocation(line: 2, column: 16, scope: !52)
!61 = !DILocation(line: 2, column: 21, scope: !52)
!62 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AIZ3foovE1CEC2ERS0_", scope: !21, file: !1, line: 2, type: !28, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, declaration: !27, retainedNodes: !10)
!63 = !DILocalVariable(name: "this", arg: 1, scope: !62, type: !64, flags: DIFlagArtificial | DIFlagObjectPointer)
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!65 = !DILocation(line: 0, scope: !62)
!66 = !DILocalVariable(name: "in", arg: 2, scope: !62, file: !1, line: 2, type: !31)
!67 = !DILocation(line: 2, column: 8, scope: !62)
!68 = !DILocation(line: 2, column: 14, scope: !62)
!69 = !DILocation(line: 2, column: 16, scope: !62)
!70 = !DILocation(line: 2, column: 21, scope: !62)
!71 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !72, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!72 = !DISubroutineType(types: !73)
!73 = !{!13}
!74 = !DILocation(line: 8, column: 5, scope: !7, inlinedAt: !75)
!75 = distinct !DILocation(line: 17, column: 14, scope: !71)
!76 = !DILocation(line: 9, column: 8, scope: !7, inlinedAt: !75)
!77 = !DILocation(line: 12, column: 7, scope: !47, inlinedAt: !75)
!78 = !DILocation(line: 13, column: 10, scope: !47, inlinedAt: !75)
!79 = !DILocation(line: 17, column: 21, scope: !71)
