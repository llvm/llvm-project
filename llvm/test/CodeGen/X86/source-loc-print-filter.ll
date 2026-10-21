; RUN: llc -mtriple=x86_64 -O2 -print-after-all \
; RUN:   -filter-print-source-locs=source.c:10 -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=MATCH

; RUN: llc -mtriple=x86_64 -O2 -print-after-all \
; RUN:   -filter-print-funcs=* -filter-print-source-locs=source.c:10 \
; RUN:   -o /dev/null < %s 2>&1 | FileCheck %s --check-prefix=MATCH

; RUN: llc -mtriple=x86_64 -O2 -print-after-all \
; RUN:   -filter-print-source-locs=source.c:999 -o /dev/null < %s 2>&1 \
; RUN:   | FileCheck %s --allow-empty --check-prefix=EMPTY

; Check source-location filtering in the legacy change printer.
; RUN: llc -mtriple=x86_64 -O2 -filetype=null -print-changed=quiet \
; RUN:   -filter-passes=x86-isel -filter-print-source-locs=source.c:10 %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHANGED

; Check that removing the last matching machine location is reported.
; RUN: llc -mtriple=x86_64 -O2 -filetype=null -print-changed=quiet \
; RUN:   -filter-passes=peephole-opt -filter-print-source-locs=source.c:10 %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHANGED-REMOVED

; Check source-location filtering for legacy module passes.
; RUN: llc -mtriple=x86_64 -O0 -filetype=null -print-changed=quiet \
; RUN:   -filter-passes=pre-isel-intrinsic-lowering \
; RUN:   -filter-print-source-locs=source.c:30 %s 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHANGED-MODULE

; Check source-location filtering for legacy IR function passes.
; RUN: llc -mtriple=x86_64 -O0 -filetype=null -print-changed=quiet \
; RUN:   -filter-passes=atomic-expand -filter-print-source-locs=source.c:40 \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=CHANGED-IR

; RUN: not --crash llc -mtriple=x86_64 -O0 -filetype=null \
; RUN:   -print-changed=quiet -filter-passes=x86-isel \
; RUN:   -filter-print-funcs=missing -filter-print-source-locs=source.c: \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=INVALID-EMPTY-LINE

; MATCH:      IR Dump After
; MATCH:      define i32 @foo
; MATCH-NOT:  define i32 @bar
; MATCH:      Machine code for function foo
; MATCH-NOT:  Machine code for function bar
; MATCH-NOT:  Machine code for function lr
; MATCH-NOT:  Machine code for function atomic_load

; EMPTY-NOT: IR Dump After
; EMPTY-NOT: Machine code for function

; CHANGED:      *** IR Dump After X86 DAG->DAG Instruction Selection (x86-isel) on foo ***
; CHANGED:      Machine code for function foo
; CHANGED-NOT:  on bar

; CHANGED-REMOVED: *** IR Deleted After Peephole Optimizations (peephole-opt) on foo ***
; CHANGED-REMOVED-NOT: on bar

; CHANGED-MODULE:      *** IR Dump After Pre-ISel Intrinsic Lowering (pre-isel-intrinsic-lowering) on
; CHANGED-MODULE-NOT:  define i32 @foo
; CHANGED-MODULE-NOT:  define i32 @bar
; CHANGED-MODULE:      define ptr @lr

; CHANGED-IR:      *** IR Dump After Expand Atomic instructions (atomic-expand) on atomic_load ***
; CHANGED-IR-NEXT: define i128 @atomic_load
; CHANGED-IR-NOT:  define i32 @foo
; CHANGED-IR-NOT:  define i32 @bar

; INVALID-EMPTY-LINE: LLVM ERROR: Invalid -filter-print-source-locs value 'source.c:'

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

define ptr @lr(ptr %p, i32 %n) !dbg !17 {
entry:
  %result = call ptr @llvm.load.relative.i32(ptr %p, i32 %n), !dbg !18
  ret ptr %result, !dbg !19
}

declare ptr @llvm.load.relative.i32(ptr, i32)

define i128 @atomic_load(ptr %p) !dbg !20 {
entry:
  %value = load atomic i128, ptr %p seq_cst, align 16, !dbg !21
  ret i128 %value, !dbg !22
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
!17 = distinct !DISubprogram(name: "lr", scope: !1, file: !1, line: 29, type: !4, scopeLine: 29, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DILocation(line: 30, column: 7, scope: !17)
!19 = !DILocation(line: 31, column: 3, scope: !17)
!20 = distinct !DISubprogram(name: "atomic_load", scope: !1, file: !1, line: 39, type: !4, scopeLine: 39, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocation(line: 40, column: 7, scope: !20)
!22 = !DILocation(line: 41, column: 3, scope: !20)
