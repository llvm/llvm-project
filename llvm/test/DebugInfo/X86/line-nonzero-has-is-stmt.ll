; Test that non-zero line records DO have is_stmt set
; This is for comparison with line 0 behavior (issue #33870 / Bugzilla #34522)
;
; When scopeLine is non-zero, the initial location directive should have is_stmt set.

; RUN: %llc_dwarf -mtriple=x86_64-unknown-linux -O0 -filetype=obj < %s | llvm-dwarfdump --debug-line - | FileCheck %s

; With scopeLine=10, we should see line 10 with is_stmt and prologue_end
; CHECK: 0x{{[0-9a-f]+}} 10 {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} is_stmt prologue_end

define void @bar() !dbg !6 {
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
; scopeLine is 10 (non-zero line should have is_stmt)
!6 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 10, type: !7, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
; Line 10 location (should have is_stmt)
!9 = !DILocation(line: 10, column: 5, scope: !6)
