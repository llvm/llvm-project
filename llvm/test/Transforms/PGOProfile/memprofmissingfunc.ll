;; Tests that we get a missing memprof error for a function not in profile when
;; using -pgo-warn-missing-function.

;; Avoid failures on big-endian systems that can't read the raw profile properly
; REQUIRES: x86_64-linux

;; TODO: Use text profile inputs once that is available for memprof.

;; The raw profiles have been generated from the source used for the memprof.ll
;; test (see comments at the top of that file).

; RUN: llvm-profdata merge %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdata

; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S 2>&1 | FileCheck %s

; CHECK: memprof record not found for function hash {{.*}} _Z16funcnotinprofilev

; ModuleID = 'memprofmissingfunc.cc'
source_filename = "memprofmissingfunc.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z16funcnotinprofilev() {
entry:
  ret void
}

