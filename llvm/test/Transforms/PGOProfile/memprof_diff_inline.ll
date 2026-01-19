;; Tests that the compiler ignores smaller contexts that differ only in the
;; IsInlineFrame bool. These map to the same full context id internally, as we
;; ignore the inline frame status which may differ in feedback compiles.
;; Presumably this happens when profiles collected from different binaries are
;; merged. If we didn't pick the largest we would default them all to noncold.

;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux

;; Generate the profile and the IR.
; RUN: split-file %s %t

;; Generate indexed profile
; RUN: llvm-profdata merge %t/memprof_diff_inline.yaml -o %t.memprofdata

; RUN: opt < %t/memprof_diff_inline.ll -passes='memprof-use<profile-filename=%t.memprofdata>' -S -memprof-report-hinted-sizes -memprof-print-match-info 2>&1 | FileCheck %s --check-prefixes=MEMPROF

; MEMPROF: MemProf notcold context with id 10194276560488437434 has total profiled size 200 is matched with 1 frames
; MEMPROF: MemProf cold context with id 16342802530253093571 has total profiled size 10000 is matched with 1 frames

;--- memprof_diff_inline.yaml
---
HeapProfileRecords:
  - GUID:            _Z3foov
    AllocSites:
      # Small non-cold, full context id 16342802530253093571, should ignore
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z4foo2v, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3barv, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 8, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
      # Large cold, full context id 16342802530253093571, should keep
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z4foo2v, LineOffset: 1, Column: 10, IsInlineFrame: true }
          - { Function: _Z3barv, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 8, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       10000
          TotalLifetime:   200000
          TotalLifetimeAccessDensity: 0
      # Small non-cold, full context id 16342802530253093571, should ignore
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z4foo2v, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3barv, LineOffset: 1, Column: 10, IsInlineFrame: true }
          - { Function: main, LineOffset: 8, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       100
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
      # Small non-cold, full context id 10194276560488437434
      - Callstack:
          - { Function: _Z3foov, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z4foo2v, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: _Z3barv, LineOffset: 1, Column: 10, IsInlineFrame: false }
          - { Function: main, LineOffset: 9, Column: 13, IsInlineFrame: false }
        MemInfoBlock:
          AllocCount:      1
          TotalSize:       200
          TotalLifetime:   0
          TotalLifetimeAccessDensity: 20000
    CallSites:       []
...
;--- memprof_diff_inline.ll
; ModuleID = 'memprof_diff_inline.cc'
source_filename = "memprof_diff_inline.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.std::nothrow_t" = type { i8 }

@_ZSt7nothrow = external global %"struct.std::nothrow_t", align 1

define dso_local noundef ptr @_Z3foov() !dbg !10 {
entry:
  ; MEMPROF: call {{.*}} @_Znwm{{.*}} !memprof ![[M1:[0-9]+]], !callsite ![[C1:[0-9]+]]
  %call = call noalias noundef align 32 ptr @_Znwm(i64 noundef 32) #6, !dbg !13
  ret ptr %call
}

declare noundef ptr @_Znwm(i64 noundef)

attributes #6 = { builtin allocsize(0) }

; MEMPROF: ![[M1]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]]}

; MEMPROF: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"notcold", ![[CONTEXTSIZE1:[0-9]+]]}
; MEMPROF: ![[STACK1]] = !{i64 2732490490862098848, i64 8467819354083268568, i64 9086428284934609951, i64 2061451396820446691}
;; Full context id 10194276560488437434 == -8252467513221114182
; MEMPROF: ![[CONTEXTSIZE1]] = !{i64 -8252467513221114182, i64 200}

; MEMPROF: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"cold", ![[CONTEXTSIZE2:[0-9]+]]}
; MEMPROF: ![[STACK2]] = !{i64 2732490490862098848, i64 8467819354083268568, i64 9086428284934609951, i64 -5747251260480066785}
;; Full context id 16342802530253093571 == -2103941543456458045
;; We should have kept the large (cold) one.
; MEMPROF: ![[CONTEXTSIZE2]] = !{i64 -2103941543456458045, i64 10000}

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
