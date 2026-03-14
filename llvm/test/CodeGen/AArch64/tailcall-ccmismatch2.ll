; RUN: llc -o - %s | FileCheck %s
; RUN: llc -global-isel -verify-machineinstrs -o - %s | FileCheck %s
target triple="aarch64--"

declare void @somefunc()
define preserve_allcc void @test_ccmismatch_notail() {
; Ensure that no tail call is used here, as the called function somefunc does
; not preserve enough registers for preserve_allcc.
; CHECK-LABEL: test_ccmismatch_notail:
; CHECK-NOT: b somefunc
; CHECK: bl somefunc
  tail call void @somefunc()
  ret void
}

declare preserve_allcc void @some_preserve_all_func()
define void @test_ccmismatch_tail() {
; We can perform a tail call here, because some_preserve_all_func preserves
; all registers necessary for test_ccmismatch_tail.
; CHECK-LABEL: test_ccmismatch_tail:
; CHECK-NOT: bl some_preserve_all_func
; CHECK: b some_preserve_all_func
  tail call preserve_allcc void @some_preserve_all_func()
  ret void
}
