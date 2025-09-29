; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-info - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; inline __attribute__((always_inline))
; int removed() { static int A; return A++; }
;
; __attribute__((always_inline))
; int not_removed() { static int B; return B++; }
;
; int foo() { return removed() + not_removed(); }

; Ensure that global variables belong to the correct subprograms even if those
; subprograms are inlined.

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_abstract_origin {{.*}} "_Z11not_removedv"
; TODO: This variable should be emitted in abstract subprogram DIE.
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_name     ("B")
; CHECK:     NULL
; CHECK:   DW_TAG_base_type
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name       ("removed")
; CHECK:     DW_TAG_variable
; CHECK:       DW_AT_name     ("A")
; CHECK:     NULL
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name       ("not_removed")
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name       ("foo")
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:     DW_TAG_inlined_subroutine
; CHECK:     NULL
; CHECK:   NULL

@_ZZ11not_removedvE1A = internal global i32 0, align 4, !dbg !0
@_ZZ7removedvE1A = linkonce_odr dso_local global i32 0, align 4, !dbg !10

define dso_local i32 @_Z11not_removedv() !dbg !2 {
  %1 = load i32, i32* @_ZZ11not_removedvE1A, align 4, !dbg !24
  %2 = add nsw i32 %1, 1, !dbg !24
  store i32 %2, i32* @_ZZ11not_removedvE1A, align 4, !dbg !24
  ret i32 %1, !dbg !25
}

define dso_local i32 @_Z3foov() !dbg !26 {
  %1 = load i32, i32* @_ZZ7removedvE1A, align 4, !dbg !27
  %2 = add nsw i32 %1, 1, !dbg !27
  store i32 %2, i32* @_ZZ7removedvE1A, align 4, !dbg !27
  %3 = load i32, i32* @_ZZ11not_removedvE1A, align 4, !dbg !29
  %4 = add nsw i32 %3, 1, !dbg !29
  store i32 %4, i32* @_ZZ11not_removedvE1A, align 4, !dbg !29
  %5 = add nsw i32 %1, %3, !dbg !31
  ret i32 %5, !dbg !32
}

!llvm.dbg.cu = !{!7}
!llvm.module.flags = !{!14, !15, !16, !17, !18, !19, !20, !21, !22}
!llvm.ident = !{!23}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !3, line: 5, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "not_removed", linkageName: "_Z11not_removedv", scope: !3, file: !3, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !13)
!3 = !DIFile(filename: "example.cpp", directory: "")
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !9, splitDebugInlining: false, nameTableKind: None)
!8 = !DIFile(filename: "example.cpp", directory: "")
!9 = !{!0, !10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "A", scope: !12, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!12 = distinct !DISubprogram(name: "removed", linkageName: "_Z7removedv", scope: !3, file: !3, line: 2, type: !4, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !13)
!13 = !{}
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 1, !"branch-target-enforcement", i32 0}
!18 = !{i32 1, !"sign-return-address", i32 0}
!19 = !{i32 1, !"sign-return-address-all", i32 0}
!20 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!21 = !{i32 7, !"uwtable", i32 1}
!22 = !{i32 7, !"frame-pointer", i32 1}
!23 = !{!"clang version 14.0.0"}
!24 = !DILocation(line: 5, column: 43, scope: !2)
!25 = !DILocation(line: 5, column: 35, scope: !2)
!26 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 7, type: !4, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !13)
!27 = !DILocation(line: 2, column: 39, scope: !12, inlinedAt: !28)
!28 = distinct !DILocation(line: 7, column: 20, scope: !26)
!29 = !DILocation(line: 5, column: 43, scope: !2, inlinedAt: !30)
!30 = distinct !DILocation(line: 7, column: 32, scope: !26)
!31 = !DILocation(line: 7, column: 30, scope: !26)
!32 = !DILocation(line: 7, column: 13, scope: !26)
