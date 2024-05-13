; RUN: llc -mtriple=thumbv7-linux-gnueabi -O0 < %s | FileCheck %s
; RUN: llc -mtriple=thumbv8m.base-arm-none-eabi -filetype=obj < %s

; Primarily a non-crash test: Thumbv7 Linux does not have FastISel support,
; which led (via a convoluted route) to DAG nodes after a TC_RETURN that
; couldn't possibly work.

declare ptr @g(ptr)

define ptr @f(ptr %a) {
entry:
  %0 = tail call ptr @g(ptr %a)
  ret ptr %0
; CHECK: b g
; CHECK-NOT: ldr
; CHECK-NOT: str
}
