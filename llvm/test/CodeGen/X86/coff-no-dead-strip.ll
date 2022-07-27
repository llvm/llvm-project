; RUN: llc -mtriple i686-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ULP
; RUN: llc -mtriple x86_64-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-NOULP

@i = global i32 0
@j = weak global i32 0
@k = internal global i32 0
declare x86_vectorcallcc void @l()
@m = private global i32 0

@llvm.used = appending global [5 x ptr] [ptr @i, ptr @j, ptr @k, ptr @l, ptr @m]

; CHECK: .section .drectve
; CHECK-ULP: .ascii " /INCLUDE:_i"
; CHECK-ULP: .ascii " /INCLUDE:_j"
; CHECK-ULP-NOT: .ascii " /INCLUDE:_k"
; CHECK-ULP-NOT: .ascii " /INCLUDE:L_m"
; CHECK-NOULP: .ascii " /INCLUDE:i"
; CHECK-NOULP: .ascii " /INCLUDE:j"
; CHECK-NOULP-NOT: .ascii " /INCLUDE:k"
; CHECK-NOULP-NOT: .ascii " /INCLUDE:.Lm"
; CHECK: .ascii " /INCLUDE:l@@0"

