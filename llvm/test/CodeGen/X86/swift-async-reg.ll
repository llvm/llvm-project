; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -fast-isel | FileCheck %s

define i8* @argument(i8* swiftasync %in) {
; CHECK-LABEL: argument:
; CHECK: movq %r14, %rax

  ret i8* %in
}

define void @call(i8* %in) {
; CHECK-LABEL: call:
; CHECK: movq %rdi, %r14

  call i8* @argument(i8* swiftasync %in)
  ret void
}
