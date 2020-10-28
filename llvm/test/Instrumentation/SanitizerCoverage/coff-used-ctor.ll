; Checks that sancov.module_ctor is marked used.
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-inline-8bit-counters=1 -sanitizer-coverage-pc-table=1 -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-8bit-counters=1 -sanitizer-coverage-pc-table=1 -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

define void @foo() {
entry:
  ret void
}

; CHECK: @llvm.used = appending global {{.*}} @sancov.module_ctor
