; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -o - %s | FileCheck %s

; Most important point here is that the promotion of the i1 works
; correctly. Previously LLVM thought that i64 was the appropriate SetCC output,
; which meant it proceded in two steps and produced an i64 -> i64 any_ext which
; couldn't be selected and faulted.

; It was expecting the smallest legal promotion of i1 to be the preferred SetCC
; type, so we'll satisfy it (this actually arguably gives better code anyway,
; with flag-manipulation operations allowed to use W-registers).

declare {i64, i1} @llvm.umul.with.overflow.i64(i64, i64)

define i64 @test_select(i64 %lhs, i64 %rhs) {
; CHECK-LABEL: test_select:

  %res = call {i64, i1} @llvm.umul.with.overflow.i64(i64 %lhs, i64 %rhs)
  %flag = extractvalue {i64, i1} %res, 1
  %retval = select i1 %flag, i64 %lhs, i64 %rhs
  ret i64 %retval
; CHECK: ret
}
