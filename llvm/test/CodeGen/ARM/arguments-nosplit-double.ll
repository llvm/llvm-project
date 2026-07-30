; RUN: llc -mtriple arm-linux-gnueabi -filetype asm -o - %s | FileCheck %s
; PR4059

declare i32 @g(double)

define i32 @f(i64 %z, i32 %a, double %b) {
  %tmp = call i32 @g(double %b)
  ret i32 %tmp
}

; CHECK-LABEL: f:
; CHECK-NOT: r3

