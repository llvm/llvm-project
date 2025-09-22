; Tests that the compiler dumps call site matches upon request.
;
; The test case is generated from:
;
; // main
; // |
; // f1 (noinline)
; // |
; // f2
; // |
; // f3 (noinline)
; // |
; // new
;
; __attribute__((noinline)) char *f3() { return ::new char[4]; }
;
; static char *f2() { return f3(); }
;
; __attribute__((noinline)) static char *f1() { return f2(); }
;
; int main() {
;   f1();
;   return 0;
; }
;
; Here we expect to match two inline call stacks:
;
; - [main]
; - [f1, f2]
;
; Note that f3 is considered to be an allocation site, not a call site, because
; it directly calls new after inlining.

; REQUIRES: x86_64-linux
; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof-dump-matched-call-site.yaml -o %t/memprof-dump-matched-call-site.memprofdata
; RUN: opt < %t/memprof-dump-matched-call-site.ll -passes='memprof-use<profile-filename=%t/memprof-dump-matched-call-site.memprofdata>' -memprof-print-match-info -S 2>&1 | FileCheck %s

;--- memprof-dump-matched-call-site.yaml
---
HeapProfileRecords:
  - GUID:            main
    AllocSites:      []
    CallSites:
      - Frames:
        - { Function: main, LineOffset: 1, Column: 3, IsInlineFrame: false }
  - GUID:            _ZL2f1v
    AllocSites:      []
    CallSites:
      - Frames:
        - { Function: _ZL2f2v, LineOffset: 0, Column: 28, IsInlineFrame: true }
        - { Function: _ZL2f1v, LineOffset: 0, Column: 54, IsInlineFrame: false }
  - GUID:            _ZL2f2v
    AllocSites:      []
    CallSites:
      - Frames:
        - { Function: _ZL2f2v, LineOffset: 0, Column: 28, IsInlineFrame: true }
        - { Function: _ZL2f1v, LineOffset: 0, Column: 54, IsInlineFrame: false }
  - GUID:            _Z2f3v
    AllocSites:
      - Callstack:
          - { Function: _Z2f3v, LineOffset: 0, Column: 47, IsInlineFrame: false }
          - { Function: _ZL2f2v, LineOffset: 0, Column: 28, IsInlineFrame: true }
          - { Function: _ZL2f1v, LineOffset: 0, Column: 54, IsInlineFrame: false }
          - { Function: main, LineOffset: 1, Column: 3, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       4
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 0
    CallSites:       []
...
;--- memprof-dump-matched-call-site.ll
; CHECK: MemProf notcold context with id 3894143216621363392 has total profiled size 4 is matched with 1 frames
; CHECK: MemProf callsite match for inline call stack 4745611964195289084 10616861955219347331
; CHECK: MemProf callsite match for inline call stack 5401059281181789382

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @_Z2f3v() {
entry:
  %call = call ptr @_Znam(i64 0), !dbg !3
  ret ptr null
}

declare ptr @_Znam(i64)

define i32 @main() {
entry:
  call void @_ZL2f1v(), !dbg !7
  ret i32 0
}

define void @_ZL2f1v() {
entry:
  %call.i = call ptr @_Z2f3v(), !dbg !9
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "match.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocation(line: 11, column: 47, scope: !4)
!4 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 11, type: !5, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 18, column: 3, scope: !8)
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 17, type: !5, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!9 = !DILocation(line: 13, column: 28, scope: !10, inlinedAt: !11)
!10 = distinct !DISubprogram(name: "f2", linkageName: "_ZL2f2v", scope: !1, file: !1, line: 13, type: !5, scopeLine: 13, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = distinct !DILocation(line: 15, column: 54, scope: !12)
!12 = distinct !DISubprogram(name: "f1", linkageName: "_ZL2f1v", scope: !1, file: !1, line: 15, type: !13, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DISubroutineType(cc: DW_CC_nocall, types: !6)
