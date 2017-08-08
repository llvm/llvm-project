; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

; Transform "a == C ? C : x" to "a == C ? a : x" to avoid materializing C.
; CHECK-LABEL: test1:
; CHECK: cmp w[[REG1:[0-9]+]], #2
; CHECK: orr w[[REG2:[0-9]+]], wzr, #0x7
; CHECK: csel w0, w[[REG1]], w[[REG2]], eq
define i32 @test1(i32 %x) {
  %cmp = icmp eq i32 %x, 2
  %res = select i1 %cmp, i32 2, i32 7
  ret i32 %res
}

; Transform "a == C ? C : x" to "a == C ? a : x" to avoid materializing C.
; CHECK-LABEL: test2:
; CHECK: cmp x[[REG1:[0-9]+]], #2
; CHECK: orr w[[REG2:[0-9]+]], wzr, #0x7
; CHECK: csel x0, x[[REG1]], x[[REG2]], eq
define i64 @test2(i64 %x) {
  %cmp = icmp eq i64 %x, 2
  %res = select i1 %cmp, i64 2, i64 7
  ret i64 %res
}

; Transform "a != C ? x : C" to "a != C ? x : a" to avoid materializing C.
; CHECK-LABEL: test3:
; CHECK: cmp x[[REG1:[0-9]+]], #7
; CHECK: orr w[[REG2:[0-9]+]], wzr, #0x2
; CHECK: csel x0, x[[REG2]], x[[REG1]], ne
define i64 @test3(i64 %x) {
  %cmp = icmp ne i64 %x, 7
  %res = select i1 %cmp, i64 2, i64 7
  ret i64 %res
}

; Don't transform "a == C ? C : x" to "a == C ? a : x" if a == 0.  If we did we
; would needlessly extend the live range of x0 when we can just use xzr.
; CHECK-LABEL: test4:
; CHECK: cmp x0, #0
; CHECK: orr w8, wzr, #0x7
; CHECK: csel x0, xzr, x8, eq
define i64 @test4(i64 %x) {
  %cmp = icmp eq i64 %x, 0
  %res = select i1 %cmp, i64 0, i64 7
  ret i64 %res
}

; Don't transform "a == C ? C : x" to "a == C ? a : x" if a == 1.  If we did we
; would needlessly extend the live range of x0 when we can just use xzr with
; CSINC to materialize the 1.
; CHECK-LABEL: test5:
; CHECK: cmp x0, #1
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x7
; CHECK: csinc x0, x[[REG]], xzr, ne
define i64 @test5(i64 %x) {
  %cmp = icmp eq i64 %x, 1
  %res = select i1 %cmp, i64 1, i64 7
  ret i64 %res
}

; Don't transform "a == C ? C : x" to "a == C ? a : x" if a == -1.  If we did we
; would needlessly extend the live range of x0 when we can just use xzr with
; CSINV to materialize the -1.
; CHECK-LABEL: test6:
; CHECK: cmn x0, #1
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x7
; CHECK: csinv x0, x[[REG]], xzr, ne
define i64 @test6(i64 %x) {
  %cmp = icmp eq i64 %x, -1
  %res = select i1 %cmp, i64 -1, i64 7
  ret i64 %res
}

; CHECK-LABEL: test7:
; CHECK: cmp x[[REG:[0-9]]], #7
; CHECK: csinc x0, x[[REG]], xzr, eq
define i64 @test7(i64 %x) {
  %cmp = icmp eq i64 %x, 7
  %res = select i1 %cmp, i64 7, i64 1
  ret i64 %res
}

; CHECK-LABEL: test8:
; CHECK: cmp x[[REG:[0-9]]], #7
; CHECK: csinc x0, x[[REG]], xzr, eq
define i64 @test8(i64 %x) {
  %cmp = icmp ne i64 %x, 7
  %res = select i1 %cmp, i64 1, i64 7
  ret i64 %res
}

; CHECK-LABEL: test9:
; CHECK: cmp x[[REG:[0-9]]], #7
; CHECK: csinv x0, x[[REG]], xzr, eq
define i64 @test9(i64 %x) {
  %cmp = icmp eq i64 %x, 7
  %res = select i1 %cmp, i64 7, i64 -1
  ret i64 %res
}

; Rather than use a CNEG, use a CSINV to transform "a == 1 ? 1 : -1" to
; "a == 1 ? a : -1" to avoid materializing a constant.
; CHECK-LABEL: test10:
; CHECK: cmp w[[REG:[0-9]]], #1
; CHECK: csinv w0, w[[REG]], wzr, eq
define i32 @test10(i32 %x) {
  %cmp = icmp eq i32 %x, 1
  %res = select i1 %cmp, i32 1, i32 -1
  ret i32 %res
}
