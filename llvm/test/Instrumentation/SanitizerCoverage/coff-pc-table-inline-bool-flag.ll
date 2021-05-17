; Checks that the PC and 8-bit Counter Arrays are placed in their own sections in COFF binaries.
; RUN: opt < %s -sancov -sanitizer-coverage-level=1 -sanitizer-coverage-inline-bool-flag=1 -sanitizer-coverage-pc-table=1 -S -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-inline-bool-flag=1 -sanitizer-coverage-pc-table=1 -S | FileCheck %s
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26433"

define void @foo() {
entry:
  ret void
}

; CHECK-DAG: section ".SCOV{{\$}}BM",
; CHECK-DAG: section ".SCOVP{{\$}}M",
; CHECK:     @__start___sancov_bools = external hidden global i1
; CHECK:     @__stop___sancov_bools = external hidden global i1
