; REQUIRES: asserts
; Force the size to be small to check assertion message.
; RUN: not --crash opt -S %s -O2 -o - -non-global-value-max-name-size=0 2>&1 | FileCheck %s
; CHECK: Can't generate unique name: MaxNameSize is too small.

define i32 @f(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  %d = add i32 %c, %a
  %e = add i32 %d, %b
  ret i32 %e
}
