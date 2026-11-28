;; Tests that an !flmd attachment must have the first Scope be the DISubprogram
;; corresponding to its parent function, contain no other DISubprograms, and
;; have all scopes point to that DISubprogram.
; RUN: llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: An MDFunctionLocalContext that is not NonDebug must contain its function's subprogram as the first scope
; CHECK-NEXT: !MDFunctionLocalContext()

; CHECK: The first scope in an MDFunctionLocalContext with debug info must be the function DISubprogram
; CHECK-NEXT: !MDFunctionLocalContext(
; CHECK-NEXT: !DILexicalBlock

; CHECK: First scope in function-local context should be subprogram pointing at the function
; CHECK-NEXT: !MDFunctionLocalContext(
; CHECK-NEXT: ptr @bar
; CHECK: First scope in function-local context should be subprogram pointing at the function
; CHECK-NEXT: !MDFunctionLocalContext(
; CHECK-NEXT: ptr @baz

; CHECK: DISubprogram should only appear as the first scope in a function-local context
; CHECK-NEXT: distinct !DISubprogram(name: "beez",

; CHECK: scope in function-local context points at wrong subprogram for function
; CHECK-NEXT: distinct !DILexicalBlock(
; CHECK-NEXT: distinct !DISubprogram(name: "bozo",


define i32 @foo(i32 %i) !dbg !9 !flmd !12 {
entry:
  ret i32 3
}

define i32 @bar() !dbg !13 !flmd !15 {
entry:
  ret i32 4
}

define i32 @baz() !dbg !16 !flmd !17 {
entry:
  ret i32 5
}

define i32 @beez() !dbg !18 !flmd !19 {
entry:
  ret i32 6
}

define i32 @bozo() !dbg !20 !flmd !21 {
entry:
  ret i32 7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 8, !"PIC Level", i32 2}
!5 = !{i32 7, !"PIE Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 24.0.0git"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = distinct !MDFunctionLocalContext(scopes: [])
!13 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 12, type: !10, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = distinct !DILexicalBlock(scope: !16, file: !1, line: 13)
!15 = distinct !MDFunctionLocalContext(scopes: [
  !!scope(scope: !14)
])
!16 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 22, type: !10, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!17 = distinct !MDFunctionLocalContext(scopes: [
  !!scope(scope: !13)
])
!18 = distinct !DISubprogram(name: "beez", scope: !1, file: !1, line: 32, type: !10, scopeLine: 32, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!19 = distinct !MDFunctionLocalContext(scopes: [
  !!scope(scope: !18),
  !!scope(scope: !18)
])
!20 = distinct !DISubprogram(name: "bozo", scope: !1, file: !1, line: 42, type: !10, scopeLine: 42, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!21 = distinct !MDFunctionLocalContext(scopes: [
  !!scope(scope: !20),
  !!scope(scope: !14)
])
