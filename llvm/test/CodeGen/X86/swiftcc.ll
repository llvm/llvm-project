; RUN: llc -mtriple x86_64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s

define swiftcc void @f() {
  %1 = alloca i8
  ret void
}

; CHECK-LABEL: f
; CHECK: .seh_stackalloc 8
; CHECK: .seh_endprologue

