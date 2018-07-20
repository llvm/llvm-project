; RUN: llc -mtriple thumbv7-windows-msvc -filetype asm -o - %s | FileCheck %s

@i = global i32 0
@j = weak global i32 0
@k = internal global i32 0

@llvm.used = appending global [3 x i8*] [i8* bitcast (i32* @i to i8*), i8* bitcast (i32* @j to i8*), i8* bitcast (i32* @k to i8*)]

; CHECK: .section .drectve
; CHECK: .ascii " /INCLUDE:i"
; CHECK: .ascii " /INCLUDE:j"
; CHECK-NOT: .ascii " /INCLUDE:k"

