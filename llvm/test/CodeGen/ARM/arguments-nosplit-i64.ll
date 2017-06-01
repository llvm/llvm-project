; RUN: llc -mtriple arm-linux-gnueabi -filetype asm -o - %s | FileCheck %s
; PR4058

declare i32 @g(i64)

define i32 @f(i64 %z, i32 %a, i64 %b) {
  %tmp = call i32 @g(i64 %b)
  ret i32 %tmp
}

; CHECK-NOT: r3

