; RUN: not llc < %s -mtriple=xcore -mcpu=xs1b-generic 2>&1 | FileCheck %s

; XCoreLowerThreadLocal expands a thread-local global into an array with one
; element per hardware thread, which requires a known per-element size. A
; zero-length array value type makes that impossible, and should be rejected
; with a clear diagnostic rather than producing a malformed GEP that only
; fails much later (and much less clearly) in instruction selection.

@G = external thread_local global [0 x i32]
; CHECK: Size of thread local object 'G' is unknown

define ptr @f() {
  ret ptr @G
}
