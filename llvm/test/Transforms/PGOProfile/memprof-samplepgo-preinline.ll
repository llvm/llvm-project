;; Test that when using both a sample profile and a memory profile, the
;; pre-inliner (which is run to stabilize the inlining seen by MemProfUsePass)
;; ignores the sample profile by default (-memprof-preinline-ignore-profile).
;;
;; The sample profile makes _Z3barv's entry hot. If the pre-inliner used the
;; sample profile (-memprof-preinline-ignore-profile=false), that would raise
;; its threshold for calls to _Z3barv, so it would be inlined into _Z3foov
;; before memprof matching. If the binary the memory profile was collected from
;; did not inline _Z3barv (e.g. because it was built with a different sample
;; profile revision in which _Z3barv was not hot), the profile attaches the
;; allocation context only to _Z3barv's record, and the copy of the allocation
;; inlined into _Z3foov is left without a hint.
;;
;; Original C++ code:
;; 1 char *bar() {
;; 2   return new char[10];
;; 3 }
;; 4 char *foo() {
;; 5   return bar();
;; 6 }

; REQUIRES: x86_64-linux

; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof.yaml -o %t/memprof.memprofdata

; RUN: opt < %t/t.ll -passes='thinlto-pre-link<O2>' \
; RUN:   -pgo-kind=pgo-sample-use-pipeline -profile-file=%t/sample.prof \
; RUN:   -memory-profile-file=%t/memprof.memprofdata \
; RUN:   -pass-remarks=inline -pass-remarks-missed=inline -S 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,IGNORE_PROF

;; Ensure that forcing the inliner cost-benefit analysis, which requires the
;; profile summary, works when ignoring the profile.
; RUN: opt < %t/t.ll -passes='thinlto-pre-link<O2>' \
; RUN:   -pgo-kind=pgo-sample-use-pipeline -profile-file=%t/sample.prof \
; RUN:   -memory-profile-file=%t/memprof.memprofdata \
; RUN:   -inline-enable-cost-benefit-analysis=true \
; RUN:   -pass-remarks=inline -pass-remarks-missed=inline -S 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,IGNORE_PROF

; RUN: opt < %t/t.ll -passes='thinlto-pre-link<O2>' \
; RUN:   -pgo-kind=pgo-sample-use-pipeline -profile-file=%t/sample.prof \
; RUN:   -memory-profile-file=%t/memprof.memprofdata \
; RUN:   -memprof-preinline-ignore-profile=false \
; RUN:   -pass-remarks=inline -pass-remarks-missed=inline -S 2>&1 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,PROFILE

;; When using the profile, the hot callee is inlined by the pre-inliner.
; PROFILE: remark: t.cc:5:10: '_Z3barv' inlined into '_Z3foov' with (cost=200, threshold={{[0-9]+}})
;; When ignoring the profile (the default) it isn't inlined by the pre-inliner
;; (nor later by the inliner, which uses a 0 threshold for hot callsites in the
;; pre-link).
; IGNORE_PROF: remark: t.cc:5:10: '_Z3barv' not inlined into '_Z3foov' because too costly to inline (cost=200, threshold={{[0-9]+}})
; IGNORE_PROF-NOT: '_Z3barv' inlined into '_Z3foov'

;; The out-of-line copy of the allocation is always hinted.
; CHECK-LABEL: define {{.*}} @_Z3barv()
; CHECK: call {{.*}} @_Znam(i64 10) #[[BARATTR:[0-9]+]]

;; When using the profile, the allocation inlined into _Z3foov does not match
;; the memory profile and is not hinted.
; CHECK-LABEL: define {{.*}} @_Z3foov()
; PROFILE: call {{.*}} @_Znam(i64 10) #[[FOOATTR:[0-9]+]]
; IGNORE_PROF: call {{.*}} @_Z3barv()

; CHECK: attributes #[[BARATTR]] = { builtin allocsize(0) "memprof"="cold" }
; PROFILE: attributes #[[FOOATTR]] = { builtin allocsize(0) }

;--- memprof.yaml
---
HeapProfileRecords:
  - GUID:            _Z3barv
    AllocSites:
      - Callstack:
          - { Function: _Z3barv, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
        MemInfoBlock:
          TotalSize:                  400
          AllocCount:                 1
          TotalLifetimeAccessDensity: 1
          TotalLifetime:              1000000
    CallSites:       []
  - GUID:            _Z3foov
    AllocSites:      []
    CallSites:
      - Frames:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
...
;--- sample.prof
_Z3foov:200000:10000
 1: 10000 _Z3barv:10000
_Z3barv:200000:10000
 1: 10000
;--- t.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local ptr @_Z3barv() #0 !dbg !7 {
entry:
  %call = call ptr @_Znam(i64 10) #1, !dbg !10
  ret ptr %call, !dbg !11
}

define dso_local ptr @_Z3foov() #2 !dbg !12 {
entry:
  %call = call ptr @_Z3barv(), !dbg !13
  ret ptr %call, !dbg !14
}

declare ptr @_Znam(i64)

attributes #0 = { "use-sample-profile" "function-inline-cost"="200" }
attributes #1 = { builtin allocsize(0) }
attributes #2 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, debugInfoForProfiling: true)
!1 = !DIFile(filename: "t.cc", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 2, column: 10, scope: !7)
!11 = !DILocation(line: 2, column: 3, scope: !7)
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!13 = !DILocation(line: 5, column: 10, scope: !12)
!14 = !DILocation(line: 5, column: 3, scope: !12)
