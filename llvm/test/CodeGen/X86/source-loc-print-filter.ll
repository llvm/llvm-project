; RUN: llc -mtriple=x86_64 -O2 -print-after-all \
; RUN:   -filter-print-source-locs=source.c:10 -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MATCH

; RUN: llc -mtriple=x86_64 -O2 -print-after-all \
; RUN:   -filter-print-funcs=* -filter-print-source-locs=source.c:10 \
; RUN:   -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=MATCH

; RUN: llc -mtriple=x86_64 -O2 -print-after-all \
; RUN:   -filter-print-source-locs=source.c:999 -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck %s --allow-empty --check-prefix=EMPTY

; MATCH:      IR Dump After
; MATCH:      define i32 @foo
; MATCH-NOT:  define i32 @bar
; MATCH:      Machine code for function foo
; MATCH-NOT:  Machine code for function bar

; EMPTY-NOT: IR Dump After
; EMPTY-NOT: Machine code for function

define i32 @foo() !dbg !5 {
entry:
  %sum = add i32 1, 2, !dbg !10
  ret i32 %sum, !dbg !11
}

define i32 @bar() !dbg !12 {
entry:
  %sum = add i32 3, 4, !dbg !13
  ret i32 %sum, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "source.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !15)
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 9, type: !4, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 10, column: 7, scope: !5)
!11 = !DILocation(line: 11, column: 3, scope: !5)
!12 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 19, type: !4, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DILocation(line: 20, column: 7, scope: !12)
!14 = !DILocation(line: 21, column: 3, scope: !12)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
