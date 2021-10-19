; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - | FileCheck %s
; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - -fast-isel | FileCheck %s

define i8* @argument(i8* swiftasync %in) {
  ret i8* %in
}

; CHECK-LABEL: argument:
; CHECK: movq    %r14, %rax

define void @call(i8* %in) {
  call i8* @argument(i8* swiftasync %in)
  ret void
}

; CHECK-LABEL: call:
; CHECK: movq %rcx, %r14
