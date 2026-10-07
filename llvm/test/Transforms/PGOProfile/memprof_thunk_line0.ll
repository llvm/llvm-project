;; Test that a call whose debug location has line 0 is matched to the profile.
;;
;; clang gives the forwarding call in a C++ non-virtual thunk (a DISubprogram
;; with DIFlagThunk) a DILocation with line 0, and the profiler records the
;; thunk frame with line offset 0. Previously the compiler computed the offset
;; as (0 - <subprogram line>) & 0xffff, so the thunk's call never received
;; !callsite metadata and, once the method was inlined into the thunk, the
;; thunk's private copy of the call chain could not be redirected to the cold
;; clone during context disambiguation.
;;
;; The test case is generated from (with -O2 -gmlt -fdebug-info-for-profiling):
;;
;; struct A { virtual ~A(); virtual void f(); long a; };
;; struct B { virtual ~B(); virtual void g(); long b; };
;; struct C : A, B { void f() override; void g() override; };
;; void C::g() {
;;   b = (long)new char[10];
;; }
;; void viaB(B *p) { p->g(); } // calls _ZThn16_N1C1gEv -> _ZN1C1gEv (cold)
;; int main() {
;;   C c; c.g();               // calls _ZN1C1gEv directly (notcold)
;;   ...
;; }

; REQUIRES: x86_64-linux
; RUN: split-file %s %t
; RUN: llvm-profdata merge %t/memprof_thunk_line0.yaml -o %t/memprof_thunk_line0.memprofdata
; RUN: opt < %t/memprof_thunk_line0.ll -passes='memprof-use<profile-filename=%t/memprof_thunk_line0.memprofdata>' -memprof-print-match-info -S 2>&1 | FileCheck %s

;--- memprof_thunk_line0.yaml
---
HeapProfileRecords:
  - GUID:            _ZN1C1gEv
    AllocSites:
      - Callstack:
          - { Function: _ZN1C1gEv, LineOffset: 1, Column: 13, IsInlineFrame: false }
          - { Function: _ZThn16_N1C1gEv, LineOffset: 0, Column: 0, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: _ZN1C1gEv, LineOffset: 1, Column: 13, IsInlineFrame: false }
          - { Function: main, LineOffset: 1, Column: 10, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
    CallSites:       []
  - GUID:            _ZThn16_N1C1gEv
    AllocSites:      []
    CallSites:
      - Frames:
        - { Function: _ZThn16_N1C1gEv, LineOffset: 0, Column: 0, IsInlineFrame: false }
  - GUID:            main
    AllocSites:      []
    CallSites:
      - Frames:
        - { Function: main, LineOffset: 1, Column: 10, IsInlineFrame: false }
...
;--- memprof_thunk_line0.ll
;; Stack ids below:
;;   5301324431836061490  = frame _ZN1C1gEv:1:13 (the allocation)
;;   7590385790444870038  = frame main:1:10
;;  -2980577181169989635  = frame _ZThn16_N1C1gEv:0:0 (15466166892539561981 unsigned)

;; From -memprof-print-match-info: both allocation contexts and both callsites,
;; including the thunk's line 0 call, are matched.
; CHECK: MemProf cold context with id 7723570926052210122 has total profiled size 10 is matched with 1 frames
; CHECK: MemProf notcold context with id 9778334180364273387 has total profiled size 10 is matched with 1 frames
; CHECK: MemProf callsite match for inline call stack 7590385790444870038
; CHECK: MemProf callsite match for inline call stack 15466166892539561981

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Both contexts match the allocation, one notcold (from main) and one cold
;; (through the thunk).
; CHECK-LABEL: define void @_ZN1C1gEv
; CHECK: call ptr @_Znam(i64 10){{.*}} !memprof ![[M:[0-9]+]], !callsite ![[C:[0-9]+]]
define void @_ZN1C1gEv(ptr %this) !dbg !10 {
entry:
  %call = call ptr @_Znam(i64 10), !dbg !13
  ret void
}

declare ptr @_Znam(i64)

;; The thunk's forwarding call has a line 0 location. It must be matched to the
;; profile's callsite record for the thunk, which has line offset 0.
; CHECK-LABEL: define void @_ZThn16_N1C1gEv
; CHECK: call void @_ZN1C1gEv(ptr %p){{.*}} !callsite ![[CTHUNK:[0-9]+]]
define void @_ZThn16_N1C1gEv(ptr %this) !dbg !14 {
entry:
  %p = getelementptr i8, ptr %this, i64 -16
  call void @_ZN1C1gEv(ptr %p), !dbg !15
  ret void
}

; CHECK-LABEL: define i32 @main
; CHECK: call void @_ZN1C1gEv(ptr %c){{.*}} !callsite ![[CMAIN:[0-9]+]]
define i32 @main() !dbg !16 {
entry:
  %c = alloca i8, i64 32
  call void @_ZN1C1gEv(ptr %c), !dbg !17
  ret i32 0
}

; CHECK: ![[M]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]]}
; CHECK: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"notcold"}
; CHECK: ![[STACK1]] = !{i64 5301324431836061490, i64 7590385790444870038}
; CHECK: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"cold"}
; CHECK: ![[STACK2]] = !{i64 5301324431836061490, i64 -2980577181169989635}
; CHECK: ![[C]] = !{i64 5301324431836061490}
; CHECK: ![[CTHUNK]] = !{i64 -2980577181169989635}
; CHECK: ![[CMAIN]] = !{i64 7590385790444870038}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "thunk.cc", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "g", linkageName: "_ZN1C1gEv", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 13, scope: !10)
!14 = distinct !DISubprogram(linkageName: "_ZThn16_N1C1gEv", scope: !1, file: !1, line: 4, type: !11, flags: DIFlagArtificial | DIFlagThunk | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 0, scope: !14)
!16 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !1, file: !1, line: 8, type: !11, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = !DILocation(line: 9, column: 10, scope: !16)
