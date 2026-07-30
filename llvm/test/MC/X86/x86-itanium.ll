; RUN: llc -mtriple i686-windows-itanium -filetype asm -o - %s | FileCheck %s

@var = common global i32 0, align 4

; CHECK-NOT: .type  _var,@object

