; RUN: %llc_dwarf -mtriple=x86_64-linux -O0 -filetype=obj < %s              \
; RUN:  | llvm-dwarfdump --show-children --name=bar - \
; RUN:  | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; The only trace of bar() being inlined into baz() is a debug record, so no
; lexical scope, and no abstract DIE, is created for the inlined instance.
; Check that the function-local type D, whose DIE is created when the
; constructor of A<D> is emitted before baz() is processed, is placed into the
; concrete DIE of bar(), and that no abstract DIE is created for bar().

; Compiled from the source below. The debug record in baz() is added manually.
;
; template <typename T> struct A {
;   A(T &in) : a(in) {}
;   T a;
; };
;
; void bar() {
;   struct D { int k; };
;   D objD;
;   A<D> objA(objD);
; }
;
; void baz() {}

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("bar")
; CHECK-NOT: DW_AT_inline
; CHECK:   DW_TAG_structure_type
; CHECK:     DW_AT_name ("D")
; CHECK:     DW_TAG_member
; CHECK:     NULL
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_name ("objD")
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_name ("objA")
; CHECK:   NULL

%struct.D = type { i32 }
%struct.A = type { %struct.D }

define dso_local void @_Z3barv() !dbg !7 {
entry:
  %objD = alloca %struct.D, align 4
  %objA = alloca %struct.A, align 4
    #dbg_declare(ptr %objD, !29, !DIExpression(), !30)
    #dbg_declare(ptr %objA, !31, !DIExpression(), !32)
  call void @_ZN1AIZ3barvE1DEC2ERS0_(ptr %objA, ptr %objD), !dbg !32
  ret void, !dbg !33
}

define internal void @_ZN1AIZ3barvE1DEC2ERS0_(ptr %this, ptr %in) unnamed_addr align 2 !dbg !34 {
entry:
  %this.addr = alloca ptr, align 8
  %in.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
    #dbg_declare(ptr %this.addr, !35, !DIExpression(), !37)
  store ptr %in, ptr %in.addr, align 8
    #dbg_declare(ptr %in.addr, !38, !DIExpression(), !39)
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = load ptr, ptr %in.addr, align 8, !dbg !41
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %this1, ptr align 4 %0, i64 4, i1 false), !dbg !40
  ret void, !dbg !43
}

define dso_local void @_Z3bazv() !dbg !44 {
entry:
    #dbg_value(i32 0, !29, !DIExpression(), !46)
  ret void, !dbg !45
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<D>", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !4, templateParams: !19)
!4 = !{!5, !14}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, line: 3, baseType: !6, size: 32)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "D", scope: !7, file: !1, line: 7, size: 32, flags: DIFlagTypePassByValue, elements: !11)
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "k", scope: !6, file: !1, line: 7, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DISubprogram(name: "A", scope: !3, file: !1, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17, !18}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!18 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !6, size: 64)
!19 = !{!20}
!20 = !DITemplateTypeParameter(name: "T", type: !6)
!21 = !{i32 7, !"Dwarf Version", i32 5}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !DILocalVariable(name: "objD", scope: !7, file: !1, line: 8, type: !6)
!30 = !DILocation(line: 8, column: 5, scope: !7)
!31 = !DILocalVariable(name: "objA", scope: !7, file: !1, line: 9, type: !3)
!32 = !DILocation(line: 9, column: 8, scope: !7)
!33 = !DILocation(line: 10, column: 1, scope: !7)
!34 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AIZ3barvE1DEC2ERS0_", scope: !3, file: !1, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, declaration: !14, retainedNodes: !10)
!35 = !DILocalVariable(name: "this", arg: 1, scope: !34, type: !36, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!37 = !DILocation(line: 0, scope: !34)
!38 = !DILocalVariable(name: "in", arg: 2, scope: !34, file: !1, line: 2, type: !18)
!39 = !DILocation(line: 2, column: 8, scope: !34)
!40 = !DILocation(line: 2, column: 14, scope: !34)
!41 = !DILocation(line: 2, column: 16, scope: !34)
!43 = !DILocation(line: 2, column: 21, scope: !34)
!44 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 12, type: !8, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!45 = !DILocation(line: 12, column: 13, scope: !44)
!46 = !DILocation(line: 8, column: 5, scope: !7, inlinedAt: !47)
!47 = distinct !DILocation(line: 12, column: 11, scope: !44)
