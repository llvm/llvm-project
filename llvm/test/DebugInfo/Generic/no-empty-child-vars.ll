; RUN: %llc_dwarf %s -o - -filetype=obj | llvm-dwarfdump - | FileCheck %s -implicit-check-not=DW_TAG
;
; Issue #46473
; XFAIL: target=sparc{{.*}}
;
; This tests that we do not create concrete variable DIEs for variables that
; have no location -- for both ways that LLVM-IR can express a variable with
; no location. It's possible to:
;   1) Omit all dbg.values and place the variable in the subprograms retained
;      nodes list,
;   2) Have a dbg.value with an undef operand, and none with "real" operands.
; Both of these should produce the same DWARF. In the two functions below
; (qux and croix) I've modified the IR to represent both scenarios.
;
; Original C, LLVM-IR modified afterwards:
;
; int foo(int bar) {
;   int baz = 12 + bar;
;   return baz;
; }
;
; int qux(int quux) {
;   int xyzzy = foo(quux);
;   return xyzzy;
; }
;
; int croix(int quux) {
;   int xyzzy = foo(quux);
;   return xyzzy;
; }
;
;; Note the implicit DW_TAG check-not in the FileCheck command line.
; CHECK: DW_TAG_compile_unit
;;
;; First subprogram is attached to the plain "foo" function in the output
;; object. It should have locations for the two variables in the function,
;; let's be non-specific as to how.
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_abstract_origin (0x{{[0-9a-f]*}} "foo")
; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_location
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_location
;
;; Abstract subprogram; should have plain variable declarations
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_name ("foo")
; CHECK:       DW_TAG_formal_parameter
; CHECK:         DW_AT_name ("bar")
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_name ("baz")
;
; CHECK:     DW_TAG_base_type
;
;; The copy of "foo" inlined into "qux" should have no children.
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_name ("qux")
; CHECK:       DW_TAG_formal_parameter
; CHECK:       DW_TAG_variable
; CHECK:       DW_TAG_inlined_subroutine
; CHECK:     NULL
;
;; Same for the copy of foo inlined into "croix"
; CHECK:     DW_TAG_subprogram
; CHECK:       DW_AT_name ("croix")
; CHECK:       DW_TAG_formal_parameter
; CHECK:       DW_TAG_variable
; CHECK:       DW_TAG_inlined_subroutine
; CHECK:     NULL

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define dso_local i32 @foo(i32 %bar) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %bar, metadata !12, metadata !DIExpression()), !dbg !14
  %add = add nsw i32 %bar, 12, !dbg !15
  call void @llvm.dbg.value(metadata i32 %add, metadata !13, metadata !DIExpression()), !dbg !14
  ret i32 %add, !dbg !16
}

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define dso_local i32 @qux(i32 %quux) local_unnamed_addr !dbg !17 {
entry:
  %add.i = add nsw i32 %quux, 12, !dbg !24
  ret i32 %add.i, !dbg !25
}

; Function Attrs: norecurse nounwind readnone uwtable willreturn
define dso_local i32 @croix(i32 %quux) local_unnamed_addr !dbg !26 {
entry:
  call void @llvm.dbg.value(metadata i32 undef, metadata !28, metadata !DIExpression()), !dbg !30
  call void @llvm.dbg.value(metadata i32 undef, metadata !12, metadata !DIExpression()), !dbg !31
  %add.i = add nsw i32 %quux, 12, !dbg !33
  call void @llvm.dbg.value(metadata i32 undef, metadata !13, metadata !DIExpression()), !dbg !31
  call void @llvm.dbg.value(metadata i32 undef, metadata !29, metadata !DIExpression()), !dbg !30
  ret i32 %add.i, !dbg !34
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "bar", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 2, column: 16, scope: !7)
!16 = !DILocation(line: 3, column: 3, scope: !7)
!17 = distinct !DISubprogram(name: "qux", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!18 = !{!19, !20}
!19 = !DILocalVariable(name: "quux", arg: 1, scope: !17, file: !1, line: 6, type: !10)
!20 = !DILocalVariable(name: "xyzzy", scope: !17, file: !1, line: 7, type: !10)
!21 = !DILocation(line: 0, scope: !17)
!22 = !DILocation(line: 0, scope: !7, inlinedAt: !23)
!23 = distinct !DILocation(line: 7, column: 15, scope: !17)
!24 = !DILocation(line: 2, column: 16, scope: !7, inlinedAt: !23)
!25 = !DILocation(line: 8, column: 3, scope: !17)
!26 = distinct !DISubprogram(name: "croix", scope: !1, file: !1, line: 11, type: !8, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !27)
!27 = !{!28, !29}
!28 = !DILocalVariable(name: "quux", arg: 1, scope: !26, file: !1, line: 11, type: !10)
!29 = !DILocalVariable(name: "xyzzy", scope: !26, file: !1, line: 12, type: !10)
!30 = !DILocation(line: 0, scope: !26)
!31 = !DILocation(line: 0, scope: !7, inlinedAt: !32)
!32 = distinct !DILocation(line: 12, column: 15, scope: !26)
!33 = !DILocation(line: 2, column: 16, scope: !7, inlinedAt: !32)
!34 = !DILocation(line: 13, column: 3, scope: !26)
