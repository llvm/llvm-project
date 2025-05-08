; Tests that the compiler dumps an allocation site with multiple inlined frames.
;
; The test case is generated from:
;
; // main
; // |
; // f1 (noinline)
; // |
; // f2
; // |
; // f3
; // |
; // new
;
; char *f1() { return new char[3]; }
; char *f2() { return f1(); }
; __attribute__((noinline)) char *f3() { return f2(); }
;
; int main() {
;   f3();
;   return 0;
; }
;
; Here we expect to match the allocation site to encompass 3 frames.

; REQUIRES: x86_64-linux
; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof-dump-matched-alloc-site.yaml -o %t/memprof-dump-matched-alloc-site.memprofdata
; RUN: opt < %t/memprof-dump-matched-alloc-site.ll -passes='memprof-use<profile-filename=%t/memprof-dump-matched-alloc-site.memprofdata>' -memprof-print-match-info -S 2>&1 | FileCheck %s

;--- memprof-dump-matched-alloc-site.yaml
---
HeapProfileRecords:
  - GUID:            _Z2f3v
    AllocSites:
      - Callstack:
          - { Function: _ZL2f1v, LineOffset: 0, Column: 35, IsInlineFrame: true }
          - { Function: _ZL2f2v, LineOffset: 0, Column: 35, IsInlineFrame: true }
          - { Function: _Z2f3v, LineOffset: 0, Column: 47, IsInlineFrame: false }
          - { Function: main, LineOffset: 1, Column: 3, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       3
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 0
    CallSites:
      # Kept empty here because this section is irrelevant for this test.
...
;--- memprof-dump-matched-alloc-site.ll
; CHECK: MemProf notcold context with id 12978026349401156968 has total profiled size 3 is matched with 3 frames

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_Z2f3v() {
entry:
  %call.i.i = call ptr @_Znam(i64 0), !dbg !3
  ret ptr null
}

declare ptr @_Znam(i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "memprof-dump-matched-alloc-site.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocation(line: 1, column: 35, scope: !4, inlinedAt: !7)
!4 = distinct !DISubprogram(name: "f1", linkageName: "_ZL2f1v", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = distinct !DILocation(line: 2, column: 35, scope: !8, inlinedAt: !9)
!8 = distinct !DISubprogram(name: "f2", linkageName: "_ZL2f2v", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = distinct !DILocation(line: 3, column: 47, scope: !10)
!10 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocation(line: 6, column: 3, scope: !12)
!12 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !5, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
