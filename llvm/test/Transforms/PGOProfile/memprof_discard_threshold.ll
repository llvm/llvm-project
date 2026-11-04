;; Tests option to discard small noncold contexts.

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; Generate the profile and the IR.
; RUN: split-file %s %t

;; Generate indexed profile
; RUN: llvm-profdata merge %t/memprof_discard_threshold.yaml -o %t.memprofdata

;; Test default (threshold 100%). We should get the same results with and
;; without -memprof-keep-all-not-cold-contexts.

; RUN: opt < %t/memprof_discard_threshold.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -S 2>&1 | FileCheck %s --check-prefixes=CALL,MEMPROF

;; Test discarding (threshold 80%). We should get the same results with and
;; without -memprof-keep-all-not-cold-contexts.

; RUN: opt < %t/memprof_discard_threshold.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -S -memprof-report-hinted-sizes -memprof-keep-all-not-cold-contexts 2>&1 | FileCheck %s --check-prefixes=CALL,MEMPROF,REPORT

; RUN: opt < %t/memprof_discard_threshold.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -memprof-callsite-cold-threshold=80 -S 2>&1 | FileCheck %s --check-prefixes=CALL,MEMPROF80

; RUN: opt < %t/memprof_discard_threshold.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -S -memprof-callsite-cold-threshold=80 -memprof-report-hinted-sizes -memprof-keep-all-not-cold-contexts 2>&1 | FileCheck %s --check-prefixes=CALL,MEMPROF80,REPORT80

;; One context should have been discarded, with exactly 80-20 behavior.
; REPORT80: MemProf hinting: Total size for discarded non-cold full allocation context hash 7175328747938231822 for 80.00% cold bytes: 20

;--- memprof_discard_threshold.yaml
---
HeapProfileRecords:
  - GUID:            A
    AllocSites:
      - Callstack:
          - { Function: A, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: B, LineOffset: 6, Column: 13, IsInlineFrame: false }
          - { Function: C, LineOffset: 7, Column: 11, IsInlineFrame: false }
	# Cold
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       100
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: A, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: B, LineOffset: 6, Column: 13, IsInlineFrame: false }
          - { Function: D, LineOffset: 8, Column: 20, IsInlineFrame: false }
	# Not cold
	# While this one is pruned without -memprof-keep-all-not-cold-contexts,
	# if we don't track the aggregate total/cold bytes correctly for
	# discarded contexts we might think that at callsite B we are more than
	# 80% cold and discard all the non-cold contexts.
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       500
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
      - Callstack:
          - { Function: A, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: B, LineOffset: 6, Column: 13, IsInlineFrame: false }
          - { Function: E, LineOffset: 5, Column: 4, IsInlineFrame: false }
          - { Function: F, LineOffset: 4, Column: 5, IsInlineFrame: false }
	# Not cold
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       30
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
      - Callstack:
          - { Function: A, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: B, LineOffset: 6, Column: 13, IsInlineFrame: false }
          - { Function: E, LineOffset: 5, Column: 4, IsInlineFrame: false }
          - { Function: G, LineOffset: 3, Column: 7, IsInlineFrame: false }
          - { Function: H, LineOffset: 2, Column: 15, IsInlineFrame: false }
	# Cold
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       80
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: A, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: B, LineOffset: 6, Column: 13, IsInlineFrame: false }
          - { Function: E, LineOffset: 5, Column: 4, IsInlineFrame: false }
          - { Function: G, LineOffset: 3, Column: 7, IsInlineFrame: false }
          - { Function: I, LineOffset: 7, Column: 16, IsInlineFrame: false }
	# Not cold
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       20
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
    CallSites:       []

;--- memprof_discard_threshold.ll
; ModuleID = 'memprof_discard_threshold.cc'
source_filename = "memprof_discard_threshold.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local noundef ptr @A() !dbg !10 {
entry:
  ; CALL: call {{.*}} @_Znam{{.*}} !memprof ![[M1:[0-9]+]], !callsite ![[C1:[0-9]+]]
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 32), !dbg !13
  ret ptr %call
}

declare noundef nonnull ptr @_Znam(i64 noundef)

; MEMPROF: ![[M1]] = !{![[MIB1:[0-9]+]],
; REPORT-SAME: ![[MIB2:[0-9]+]],
; MEMPROF-SAME: ![[MIB3:[0-9]+]], ![[MIB4:[0-9]+]]
; REPORT-SAME:	, ![[MIB5:[0-9]+]]}
; MEMPROF: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"cold"
; REPORT-SAME:  , ![[SIZE1:[0-9]+]]}
; MEMPROF: ![[STACK1]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 705268496523263927}
; REPORT: ![[SIZE1]] = !{i64 -7154823362113119138, i64 100}
; REPORT: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"notcold", ![[SIZE2:[0-9]+]]}
; REPORT: ![[STACK2]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -6015552466537626283, i64 7621414677325196405}
; REPORT: ![[SIZE2]] = !{i64 4574148894276641937, i64 30}
; MEMPROF: ![[MIB3]] = !{![[STACK3:[0-9]+]], !"cold"
; REPORT-SAME: , ![[SIZE3:[0-9]+]]}
; MEMPROF: ![[STACK3]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -6015552466537626283, i64 -2897722569788633560, i64 3206456850862191843}
; REPORT: ![[SIZE3]] = !{i64 2848517899452258040, i64 80}
; MEMPROF: ![[MIB4]] = !{![[STACK4:[0-9]+]], !"notcold"
; REPORT-SAME: , ![[SIZE4:[0-9]+]]}
; MEMPROF: ![[STACK4]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -6015552466537626283, i64 -2897722569788633560, i64 -1037739922429764316}
; REPORT: ![[SIZE4]] = !{i64 7175328747938231822, i64 20}
; REPORT: ![[MIB5]] = !{![[STACK5:[0-9]+]], !"notcold", ![[SIZE5:[0-9]+]]}
; REPORT: ![[STACK5]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -3111772584478263690}
; REPORT: ![[SIZE5]] = !{i64 8469515017747579284, i64 500}
; MEMPROF: ![[C1]] = !{i64 -2647357475745718070}

; MEMPROF80: ![[M1]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]], ![[MIB3:[0-9]+]]
; REPORT80-SAME:	, ![[MIB5:[0-9]+]]}
; MEMPROF80: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"cold"
; REPORT80-SAME:  , ![[SIZE1:[0-9]+]]}
; MEMPROF80: ![[STACK1]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 705268496523263927}
; REPORT80: ![[SIZE1]] = !{i64 -7154823362113119138, i64 100}
; MEMPROF80: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"notcold"
; REPORT80-SAME: , ![[SIZE2:[0-9]+]]}
; MEMPROF80: ![[STACK2]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -6015552466537626283, i64 7621414677325196405}
; REPORT80: ![[SIZE2]] = !{i64 4574148894276641937, i64 30}
; MEMPROF80: ![[MIB3]] = !{![[STACK3:[0-9]+]], !"cold"
; REPORT80-SAME: , ![[SIZE3:[0-9]+]]}
; MEMPROF80: ![[STACK3]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -6015552466537626283, i64 -2897722569788633560, i64 3206456850862191843}
; REPORT80: ![[SIZE3]] = !{i64 2848517899452258040, i64 80}
; REPORT80: ![[MIB5]] = !{![[STACK5:[0-9]+]], !"notcold", ![[SIZE5:[0-9]+]]}
; REPORT80: ![[STACK5]] = !{i64 -2647357475745718070, i64 869302454322824036, i64 -3111772584478263690}
; REPORT80: ![[SIZE5]] = !{i64 8469515017747579284, i64 500}
; MEMPROF80: ![[C1]] = !{i64 -2647357475745718070}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 6cbe6284d1f0a088b5c6482ae27b738f03d82fe7)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof.cc", directory: ".", checksumkind: CSK_MD5, checksum: "e8c40ebe4b21776b4d60e9632cbc13c2")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "A", linkageName: "A", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 10, scope: !10)
