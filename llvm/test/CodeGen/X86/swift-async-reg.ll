; RUN: llc -mtriple=x86_64-apple-darwin %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -fast-isel | FileCheck %s

define ptr @argument(ptr swiftasync %in) {
; CHECK-LABEL: argument:
; CHECK: movq %r14, %rax

  ret ptr %in
}

define void @call(ptr %in) {
; CHECK-LABEL: call:
; CHECK: movq %rdi, %r14

  call ptr @argument(ptr swiftasync %in)
  ret void
}
