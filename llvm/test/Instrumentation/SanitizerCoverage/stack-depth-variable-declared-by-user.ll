; Ensure that we terminate with a useful error message (instead of crash) if the
; user declares `__sancov_lowest_stack` with an unexpected type.
; RUN: not opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 \
; RUN:         -sanitizer-coverage-stack-depth -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Wrong type: i32 instead of expected i64
@__sancov_lowest_stack = thread_local global i32 0

; CHECK: error: '__sancov_lowest_stack' should not be declared by the user
