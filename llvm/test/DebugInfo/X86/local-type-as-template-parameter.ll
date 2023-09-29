; REQUIERES: system-linux
; RUN: %llc_dwarf -mtriple=x86_64-linux -O0 -filetype=obj < %s              \
; RUN:  | llvm-dwarfdump --show-children --name=foo - \
; RUN:  | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; The test ensures that AsmPrinter doesn't crashed compiling this.
; It also demostrates misplacement for a local type (see PR55680 for details).

; The test compiled from:

; template<typename T>
; struct A {
;   A(T &in) : a(in) {}
;   T a;
; };
;
; __attribute__((always_inline))
; void foo() {
;   struct B { int i; };
;   B objB;
;   A<B> objA(objB);
; }
;
; int main() {
;   foo();
; }

; Concrete out-of-line tree of foo().
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin {{.*}} "_Z3foov"

; FIXME: 'struct B' should be in the abstract tree below, not here.
; CHECK:   DW_TAG_structure_type
; CHECK:     DW_AT_name	("B")
; CHECK:     DW_TAG_member
; CHECK:     NULL
;
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objB"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objA"

; CHECK:   NULL

; Abstract tree of foo().
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name	("foo")
; CHECK:   DW_AT_inline	(DW_INL_inlined)

; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_name	("objB")
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_name	("objA")

; CHECK:   NULL

; CHECK: DW_TAG_inlined_subroutine
; CHECK:   DW_AT_abstract_origin {{.*}} "_Z3foov"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objB"
; CHECK:   DW_TAG_variable
; CHECK:     DW_AT_abstract_origin {{.*}} "objA"
; CHECK:   NULL

%struct.B = type { i32 }
%struct.A = type { %struct.B }

define dso_local void @_Z3foov() !dbg !7 {
entry:
  %objB = alloca %struct.B, align 4
  %objA = alloca %struct.A, align 4
  call void @llvm.dbg.declare(metadata ptr %objB, metadata !30, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.declare(metadata ptr %objA, metadata !32, metadata !DIExpression()), !dbg !33
  call void @_ZN1AIZ3foovE1BEC2ERS0_(ptr noundef nonnull align 4 dereferenceable(4) %objA, ptr noundef nonnull align 4 dereferenceable(4) %objB), !dbg !33
  ret void, !dbg !34
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define internal void @_ZN1AIZ3foovE1BEC2ERS0_(ptr noundef nonnull align 4 dereferenceable(4) %this, ptr noundef nonnull align 4 dereferenceable(4) %in) unnamed_addr align 2 !dbg !35 {
entry:
  %this.addr = alloca ptr, align 8
  %in.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !36, metadata !DIExpression()), !dbg !38
  store ptr %in, ptr %in.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %in.addr, metadata !39, metadata !DIExpression()), !dbg !40
  %this1 = load ptr, ptr %this.addr, align 8
  %a = getelementptr inbounds %struct.A, ptr %this1, i32 0, i32 0, !dbg !41
  %0 = load ptr, ptr %in.addr, align 8, !dbg !42
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %0, i64 4, i1 false), !dbg !41
  ret void, !dbg !43
}

define dso_local noundef i32 @main() !dbg !44 {
entry:
  %objB.i = alloca %struct.B, align 4
  %objA.i = alloca %struct.A, align 4
  call void @llvm.dbg.declare(metadata ptr %objB.i, metadata !30, metadata !DIExpression()), !dbg !47
  call void @llvm.dbg.declare(metadata ptr %objA.i, metadata !32, metadata !DIExpression()), !dbg !49
  call void @_ZN1AIZ3foovE1BEC2ERS0_(ptr noundef nonnull align 4 dereferenceable(4) %objA.i, ptr noundef nonnull align 4 dereferenceable(4) %objB.i), !dbg !49
  ret i32 0, !dbg !50
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23, !24, !25, !26, !27, !28}
!llvm.ident = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "aec7fd397e86f8655ef7f4bb4233b849")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<B>", file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !4, templateParams: !20)
!4 = !{!5, !15}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, line: 4, baseType: !6, size: 32)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", scope: !7, file: !1, line: 9, size: 32, flags: DIFlagTypePassByValue, elements: !12)
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 8, type: !8, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = !{!6}
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !6, file: !1, line: 9, baseType: !14, size: 32)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DISubprogram(name: "A", scope: !3, file: !1, line: 3, type: !16, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!16 = !DISubroutineType(types: !17)
!17 = !{null, !18, !19}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!19 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !6, size: 64)
!20 = !{!21}
!21 = !DITemplateTypeParameter(name: "T", type: !6)
!22 = !{i32 7, !"Dwarf Version", i32 5}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"wchar_size", i32 4}
!25 = !{i32 7, !"PIC Level", i32 2}
!26 = !{i32 7, !"PIE Level", i32 2}
!27 = !{i32 7, !"uwtable", i32 2}
!28 = !{i32 7, !"frame-pointer", i32 2}
!29 = !{!"clang version 15.0.0"}
!30 = !DILocalVariable(name: "objB", scope: !7, file: !1, line: 10, type: !6)
!31 = !DILocation(line: 10, column: 5, scope: !7)
!32 = !DILocalVariable(name: "objA", scope: !7, file: !1, line: 11, type: !3)
!33 = !DILocation(line: 11, column: 8, scope: !7)
!34 = !DILocation(line: 12, column: 1, scope: !7)
!35 = distinct !DISubprogram(name: "A", linkageName: "_ZN1AIZ3foovE1BEC2ERS0_", scope: !3, file: !1, line: 3, type: !16, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, declaration: !15, retainedNodes: !10)
!36 = !DILocalVariable(name: "this", arg: 1, scope: !35, type: !37, flags: DIFlagArtificial | DIFlagObjectPointer)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!38 = !DILocation(line: 0, scope: !35)
!39 = !DILocalVariable(name: "in", arg: 2, scope: !35, file: !1, line: 3, type: !19)
!40 = !DILocation(line: 3, column: 8, scope: !35)
!41 = !DILocation(line: 3, column: 14, scope: !35)
!42 = !DILocation(line: 3, column: 16, scope: !35)
!43 = !DILocation(line: 3, column: 21, scope: !35)
!44 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !45, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!45 = !DISubroutineType(types: !46)
!46 = !{!14}
!47 = !DILocation(line: 10, column: 5, scope: !7, inlinedAt: !48)
!48 = distinct !DILocation(line: 15, column: 3, scope: !44)
!49 = !DILocation(line: 11, column: 8, scope: !7, inlinedAt: !48)
!50 = !DILocation(line: 16, column: 1, scope: !44)
