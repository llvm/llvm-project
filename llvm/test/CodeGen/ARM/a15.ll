; RUN: llc -mtriple=arm -mcpu=cortex-a15 %s -o - | FileCheck %s

; CHECK: a
define i32 @a(i32 %x) {
  ret i32 %x;
}
