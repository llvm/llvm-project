; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - | FileCheck %s
; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - -fast-isel | FileCheck %s

define ptr @argument(ptr swiftasync %in) {
  ret ptr %in
}

; CHECK-LABEL: argument:
; CHECK: movq    %r14, %rax

define void @call(ptr %in) {
  call ptr @argument(ptr swiftasync %in)
  ret void
}

; CHECK-LABEL: call:
; CHECK: movq %rcx, %r14
