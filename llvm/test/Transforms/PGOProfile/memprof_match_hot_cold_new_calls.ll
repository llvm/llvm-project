;; Tests optional matching of memprof profile on call to operator new
;; with manual hot/cold hint.

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; Generate the profile and the IR.
; RUN: split-file %s %t

;; Generate indexed profile
; RUN: llvm-profdata merge %t/memprof_match_hot_cold_new_calls.yaml -o %t.memprofdata

;; By default we should not match profile on to manually hinted operator
;; new calls, because we don't currently override the manual hints anyway.
; RUN: opt < %t/memprof_match_hot_cold_new_calls.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -S 2>&1 | FileCheck %s --implicit-check-not !memprof --implicit-check-not !callsite

;; Check that we match profiles onto these manually hinted new calls
;; under the -memprof-match-hot-cold-new=true option.
; RUN: opt < %t/memprof_match_hot_cold_new_calls.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -S -memprof-match-hot-cold-new=true 2>&1 | FileCheck %s --check-prefixes=MEMPROF

;--- memprof_match_hot_cold_new_calls.yaml
---
HeapProfileRecords:
  - GUID:            _Z3foov
    AllocSites:
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 6, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 7, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z4foo2v, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3barv, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 8, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z4foo2v, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3bazv, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 9, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 2, Column: 12, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 31, Column: 15, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 2, Column: 12, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: _Z7recursej, LineOffset: 3, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 31, Column: 15, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
    CallSites:       []
...
;--- memprof_match_hot_cold_new_calls.ll
; ModuleID = 'memprof_match_hot_cold_new_calls.cc'
source_filename = "memprof_match_hot_cold_new_calls.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.std::nothrow_t" = type { i8 }

@_ZSt7nothrow = external global %"struct.std::nothrow_t", align 1

define dso_local noundef ptr @_Z3foov() !dbg !10 {
entry:
  ; MEMPROF: call {{.*}} @_Znwm{{.*}} !memprof ![[M1:[0-9]+]], !callsite ![[C1:[0-9]+]]
  %call = call noalias noundef align 32 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef 32, i64 noundef 32, ptr noundef nonnull align 1 dereferenceable(1) @_ZSt7nothrow, i8 noundef zeroext 0), !dbg !13
  ret ptr %call
}

declare noundef ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef, i64 noundef, ptr noundef nonnull align 1 dereferenceable(1), i8 noundef zeroext)

; MEMPROF: ![[M1]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]], ![[MIB3:[0-9]+]], ![[MIB4:[0-9]+]]}
; MEMPROF: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK1]] = !{i64 2732490490862098848, i64 748269490701775343}
; MEMPROF: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK2]] = !{i64 2732490490862098848, i64 2104812325165620841, i64 6281715513834610934, i64 6281715513834610934, i64 6281715513834610934, i64 1544787832369987002}
; MEMPROF: ![[MIB3]] = !{![[STACK3:[0-9]+]], !"notcold"}
; MEMPROF: ![[STACK3]] = !{i64 2732490490862098848, i64 2104812325165620841, i64 6281715513834610934, i64 6281715513834610934, i64 6281715513834610934, i64 6281715513834610934}
; MEMPROF: ![[MIB4]] = !{![[STACK4:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK4]] = !{i64 2732490490862098848, i64 8467819354083268568}
; MEMPROF: ![[C1]] = !{i64 2732490490862098848}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 6cbe6284d1f0a088b5c6482ae27b738f03d82fe7)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof.cc", directory: "/usr/local/google/home/tejohnson/llvm/tmp", checksumkind: CSK_MD5, checksum: "e8c40ebe4b21776b4d60e9632cbc13c2")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 10, scope: !10)
