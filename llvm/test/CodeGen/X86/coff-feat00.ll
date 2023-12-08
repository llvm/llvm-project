; RUN: llc -O0 -mtriple=i386-pc-win32 -filetype=asm -o - %s | FileCheck %s

define i32 @foo() {
  ret i32 0
}

; CHECK: .set @feat.00, 1
