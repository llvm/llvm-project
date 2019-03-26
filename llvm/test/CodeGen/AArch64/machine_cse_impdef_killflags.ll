; RUN: llc < %s -mtriple=aarch64-apple-ios -fast-isel -verify-machineinstrs | FileCheck %s

; Check that the kill flag is cleared between CSE'd instructions on their
; imp-def'd registers.
; The verifier would complain otherwise.
define i64 @csed-impdef-killflag(i64 %a) {
; CHECK-LABEL: csed-impdef-killflag
; CHECK-DAG:  mov    [[REG1:w[0-9]+]], #1
; CHECK-DAG:  mov    [[REG2:x[0-9]+]], #2
; CHECK-DAG:  mov    [[REG3:x[0-9]+]], #3
; CHECK-DAG:  cmp    x0, #0
; CHECK:  csel   w[[SELECT_WREG_1:[0-9]+]], wzr, [[REG1]], ne
; CHECK-DAG:  csel   [[SELECT_XREG_2:x[0-9]+]], [[REG2]], [[REG3]], ne
; CHECK:      ubfx   [[SELECT_XREG_1:x[0-9]+]], x[[SELECT_WREG_1]], #0, #32
; CHECK-NEXT: add    x0, [[SELECT_XREG_2]], [[SELECT_XREG_1]]
; CHECK-NEXT: ret

  %1 = icmp ne i64 %a, 0
  %2 = select i1 %1, i32 0, i32 1
  %3 = icmp ne i64 %a, 0
  %4 = select i1 %3, i64 2, i64 3
  %5 = zext i32 %2 to i64
  %6 = add i64 %4, %5
  ret i64 %6
}
