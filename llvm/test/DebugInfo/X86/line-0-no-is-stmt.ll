; Test that line 0 records don't have is_stmt set
; This tests the fix for LLVM issue #33870 (Bugzilla #34522)
;
; When scopeLine is 0, the initial location directive should not have is_stmt set.

; RUN: %llc_dwarf -mtriple=x86_64-unknown-linux -O0 -filetype=obj < %s | llvm-dwarfdump --debug-line - | FileCheck %s

; The line table entry for line 0 should exist but not have "is_stmt" in its Flags
; CHECK: Address
; CHECK: 0x{{[0-9a-f]+}} 0 0
; CHECK-NOT: is_stmt
; CHECK: end_sequence

define void @foo() !dbg !6 {
entry:
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{}
; scopeLine is 0 (line 0 should not have is_stmt)
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 10, type: !7, scopeLine: 0, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
; Line 0 location (should not have is_stmt)
!9 = !DILocation(line: 0, column: 0, scope: !6)
