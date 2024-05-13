; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

@i = linkonce_odr global i32 1

; CHECK: f:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: .long	1
define void @f() prologue i32 1 {
  ret void
}

; CHECK: g:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: .quad	i
define void @g() prologue ptr @i {
  ret void
}
