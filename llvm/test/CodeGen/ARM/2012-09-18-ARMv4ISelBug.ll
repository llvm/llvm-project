; RUN: llc -mtriple=arm-eabi -mcpu=arm7tdmi %s -o - | FileCheck %s

; movw is only legal for V6T2 and later.
; rdar://12300648

define i32 @t(i32 %x) {
; CHECK-LABEL: t:
; CHECK-NOT: movw
  %tmp = add i32 %x, -65535
  ret i32 %tmp
}
