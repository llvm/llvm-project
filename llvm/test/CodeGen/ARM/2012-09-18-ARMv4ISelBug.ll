; RUN: llc < %s -march=arm -mcpu=arm7tdmi | FileCheck %s

; movw is only legal for V6T2 and later.
; rdar://12300648

define i32 @t(i32 %x) {
; CHECK: t:
; CHECK-NOT: movw
  %tmp = add i32 %x, -65535
  ret i32 %tmp
}
