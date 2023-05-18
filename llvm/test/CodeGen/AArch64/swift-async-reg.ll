; RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios %s -o - -global-isel | FileCheck %s
; RUN: llc -mtriple=arm64-apple-ios %s -o - -fast-isel | FileCheck %s

define ptr @argument(ptr swiftasync %in) {
; CHECK-LABEL: argument:
; CHECK: mov x0, x22

  ret ptr %in
}

define void @call(ptr %in) {
; CHECK-LABEL: call:
; CHECK: mov x22, x0

  call ptr @argument(ptr swiftasync %in)
  ret void
}
