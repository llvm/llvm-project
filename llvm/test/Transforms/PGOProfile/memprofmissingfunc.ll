;; Tests that we get a missing memprof error for a function not in profile when
;; using -pgo-warn-missing-function.

; RUN: rm -rf %t && split-file %s %t

; RUN: llvm-profdata merge %t/a.yaml -o %t/a.memprofdata
; RUN: opt < %t/a.ll -passes='memprof-use<profile-filename=%t/a.memprofdata>' -pgo-warn-missing-function -S 2>&1 | FileCheck %s

; CHECK: memprof record not found for function hash {{.*}} _Z16funcnotinprofilev

;--- a.ll
; ModuleID = 'memprofmissingfunc.cc'
source_filename = "memprofmissingfunc.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z16funcnotinprofilev() {
entry:
  ret void
}

;--- a.yaml
---
HeapProfileRecords:
  - GUID:            main
    CallSites:
      - Frames:
          - { Function: main, LineOffset: 1, Column: 3, IsInlineFrame: false }
...
