; RUN: llc -mtriple x86_64-- -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple x86_64-unknown-windows-msvc -filetype asm -o - %s | FileCheck %s --check-prefix=MSVC

define swiftcc void @f() {
  %1 = alloca i8
  ret void
}

; MSVC-LABEL: f
; MSVC: .seh_stackalloc 8
; MSVC: .seh_endprologue

declare swiftcc i64 @myFunc()
define swiftcc i64 @myFunc2()  nounwind {
  %r = tail call swiftcc i64 @myFunc()
  ret i64 %r
}

; CHECK-LABEL: myFunc2
; CHECK: jmp myFunc
; CHECK-NOT: call
