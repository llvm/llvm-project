; RUN: llc -mtriple=x86_64-linux -fast-isel -show-mc-encoding < %s | FileCheck %s

; CHECK-LABEL: f:
; CHECK:       addl $-2, %eax         # encoding: [0x83,0xc0,0xfe]
define i32 @f(ptr %y) {
  %x = load i32, ptr %y
  %dec = add i32 %x, -2
  ret i32 %dec
}
