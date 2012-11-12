; RUN: llc < %s  -mcpu=cortex-a15 | FileCheck %s

; CHECK: a
define i32 @a(i32 %x) {
  ret i32 %x;
}
