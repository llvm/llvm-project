; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios %s -o - -global-isel | FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios %s -o - -fast-isel | FileCheck %s

define i8* @argument(i8* swiftasync %in) {
; CHECK-LABEL: argument:
; CHECK: mov x0, x22

  ret i8* %in
}

define void @call(i8* %in) {
; CHECK-LABEL: call:
; CHECK: mov x22, x0

  call i8* @argument(i8* swiftasync %in)
  ret void
}
