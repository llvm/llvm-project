; Verifies that a call site gets annotated even when a call site matches an
; allocation call stack but does not call one of the memory allocation
; functions.

; REQUIRES: x86_64-linux
; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof-call-site-at-alloc-site.yaml -o %t/memprof-call-site-at-alloc-site.memprofdata
; RUN: opt < %t/memprof-call-site-at-alloc-site.ll -passes='memprof-use<profile-filename=%t/memprof-call-site-at-alloc-site.memprofdata>' -memprof-print-match-info -S 2>&1 | FileCheck %s

;--- memprof-call-site-at-alloc-site.yaml
---
HeapProfileRecords:
  - GUID:            _Z3foov
    AllocSites:
      - Callstack:
          - { Function: _Z3foov, LineOffset: 6, Column: 12, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       898709270
          TotalLifetime:   1000000
          TotalLifetimeAccessDensity: 1
    CallSites:
      - Frames:
        - { Function: _Z3foov, LineOffset: 6, Column: 12, IsInlineFrame: false }
...

;--- memprof-call-site-at-alloc-site.ll
; CHECK: MemProf callsite match for inline call stack 774294594974568741

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

declare void @_ZN9SomethingC1Ev()

define void @_Z3foov() {
  %1 = call ptr @_Znam(), !dbg !3
  call void @_ZN9SomethingC1Ev(), !dbg !3
  ret void
}

declare ptr @_Znam()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "something.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DILocation(line: 106, column: 12, scope: !4)
!4 = distinct !DISubprogram(name: "Init", linkageName: "_Z3foov", scope: !5, file: !1, line: 100, type: !7, scopeLine: 100, spFlags: DISPFlagDefinition, unit: !0)
!5 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Something", file: !1)
!6 = !{}
!7 = distinct !DISubroutineType(types: !6)
