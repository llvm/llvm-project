; RUN: llc -mtriple thumbv7-windows-msvc -filetype asm -o - %s | FileCheck %s

@i = global i32 0
@j = weak global i32 0
@k = internal global i32 0

@llvm.used = appending global [3 x ptr] [ptr @i, ptr @j, ptr @k]

; CHECK: .section .drectve
; CHECK: .ascii " /INCLUDE:i"
; CHECK: .ascii " /INCLUDE:j"
; CHECK-NOT: .ascii " /INCLUDE:k"

