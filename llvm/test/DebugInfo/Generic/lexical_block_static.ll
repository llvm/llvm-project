; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; inline __attribute__((always_inline))
; int removed() {
;   {
;     static int A;
;     return A++;
;   }
; }
;
; __attribute__((always_inline))
; int not_removed() {
;   {
;     static int B;
;     return B++;
;   }
; }
;
; int foo() {
;   {
;     static int C;
;     return ++C + removed() + not_removed();
;   }
; }

; CHECK: DW_TAG_compile_unit

; Out-of-line definition of `not_removed()`.
; The empty lexical block is created to match abstract origin.
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_abstract_origin {{.*}} "_Z11not_removedv"
; CHECK:     DW_TAG_lexical_block
; CHECK:     NULL

; Abstract definition of `removed()`
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("removed")
; CHECK:     DW_AT_inline	(DW_INL_inlined)
; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_name	("A")
; CHECK:         DW_AT_location
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   DW_TAG_base_type

; Abstract definition of `not_removed()`
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("not_removed")
; CHECK:     DW_AT_inline	(DW_INL_inlined)
; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_name	("B")
; CHECK:         DW_AT_location
; CHECK:       NULL
; CHECK:     NULL

; Definition of foo().
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("foo")
; CHECK:     DW_TAG_lexical_block
; CHECK:       DW_TAG_inlined_subroutine
; CHECK:         DW_TAG_lexical_block
; CHECK:         NULL
; CHECK:       DW_TAG_inlined_subroutine
; CHECK:         DW_TAG_lexical_block
; CHECK:         NULL
; CHECK:       DW_TAG_variable
; CHECK:         DW_AT_name	("C")
; CHECK:         DW_AT_location
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   NULL

@_ZZ11not_removedvE1B = internal global i32 0, align 4, !dbg !0
@_ZZ3foovE1C = internal global i32 0, align 4, !dbg !10
@_ZZ7removedvE1A = linkonce_odr dso_local global i32 0, align 4, !dbg !15

define dso_local i32 @_Z11not_removedv() !dbg !4 {
entry:
  %0 = load i32, i32* @_ZZ11not_removedvE1B, align 4, !dbg !25
  %inc = add nsw i32 %0, 1, !dbg !25
  store i32 %inc, i32* @_ZZ11not_removedvE1B, align 4, !dbg !25
  ret i32 %0, !dbg !26
}

define dso_local i32 @_Z3foov() !dbg !13 {
entry:
  %0 = load i32, i32* @_ZZ3foovE1C, align 4, !dbg !27
  %inc = add nsw i32 %0, 1, !dbg !27
  store i32 %inc, i32* @_ZZ3foovE1C, align 4, !dbg !27
  %1 = load i32, i32* @_ZZ7removedvE1A, align 4, !dbg !28
  %inc.i3 = add nsw i32 %1, 1, !dbg !28
  store i32 %inc.i3, i32* @_ZZ7removedvE1A, align 4, !dbg !28
  %add = add nsw i32 %inc, %1, !dbg !30
  %2 = load i32, i32* @_ZZ11not_removedvE1B, align 4, !dbg !31
  %inc.i = add nsw i32 %2, 1, !dbg !31
  store i32 %inc.i, i32* @_ZZ11not_removedvE1B, align 4, !dbg !31
  %add2 = add nsw i32 %add, %2, !dbg !33
  ret i32 %add2, !dbg !34
}

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!19, !20, !21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !3, line: 13, type: !7, isLocal: true, isDefinition: true)
!2 = distinct !DILexicalBlock(scope: !4, file: !3, line: 12, column: 3)
!3 = !DIFile(filename: "test_static.cpp", directory: "/")
!4 = distinct !DISubprogram(name: "not_removed", linkageName: "_Z11not_removedv", scope: !3, file: !3, line: 11, type: !5, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !14)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 14.0.0 (git@github.com:llvm/llvm-project.git e1d09ac2d118825452bfc26e44565f7f4122fd6d)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !9, splitDebugInlining: false, nameTableKind: None)
!9 = !{!0, !10, !15}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "C", scope: !12, file: !3, line: 20, type: !7, isLocal: true, isDefinition: true)
!12 = distinct !DILexicalBlock(scope: !13, file: !3, line: 19, column: 3)
!13 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 18, type: !5, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !14)
!14 = !{}
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "A", scope: !17, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!17 = distinct !DILexicalBlock(scope: !18, file: !3, line: 4, column: 3)
!18 = distinct !DISubprogram(name: "removed", linkageName: "_Z7removedv", scope: !3, file: !3, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !14)
!19 = !{i32 7, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{i32 7, !"uwtable", i32 1}
!23 = !{i32 7, !"frame-pointer", i32 2}
!24 = !{!"clang version 14.0.0 (git@github.com:llvm/llvm-project.git)"}
!25 = !DILocation(line: 14, column: 13, scope: !2)
!26 = !DILocation(line: 14, column: 5, scope: !2)
!27 = !DILocation(line: 21, column: 12, scope: !12)
!28 = !DILocation(line: 6, column: 13, scope: !17, inlinedAt: !29)
!29 = distinct !DILocation(line: 21, column: 18, scope: !12)
!30 = !DILocation(line: 21, column: 16, scope: !12)
!31 = !DILocation(line: 14, column: 13, scope: !2, inlinedAt: !32)
!32 = distinct !DILocation(line: 21, column: 30, scope: !12)
!33 = !DILocation(line: 21, column: 28, scope: !12)
!34 = !DILocation(line: 21, column: 5, scope: !12)
