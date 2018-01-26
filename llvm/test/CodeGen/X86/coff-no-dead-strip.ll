; RUN: llc -mtriple i686-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-ULP
; RUN: llc -mtriple x86_64-windows-msvc -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix CHECK-NOULP

@i = global i32 0
@j = weak global i32 0
@k = internal global i32 0
declare x86_vectorcallcc void @l()
@m = private global i32 0

@llvm.used = appending global [5 x i8*] [i8* bitcast (i32* @i to i8*), i8* bitcast (i32* @j to i8*), i8* bitcast (i32* @k to i8*), i8* bitcast (void ()* @l to i8*), i8* bitcast (i32* @m to i8*)]

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

