; RUN: llc -mtriple=thumbv7-linux-gnueabi -O0 < %s | FileCheck %s
; RUN: llc -mtriple=thumbv8m.base-arm-none-eabi -filetype=obj < %s

; Primarily a non-crash test: Thumbv7 Linux does not have FastISel support,
; which led (via a convoluted route) to DAG nodes after a TC_RETURN that
; couldn't possibly work.

declare i8* @g(i8*)

define i8* @f(i8* %a) {
entry:
  %0 = tail call i8* @g(i8* %a)
  ret i8* %0
; CHECK: b g
; CHECK-NOT: ldr
; CHECK-NOT: str
}
