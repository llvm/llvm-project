; Test that interposable symbols do not get put in comdats.
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -mtriple x86_64-linux-gnu -S | FileCheck %s
; RUN: opt < %s -sancov -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -mtriple x86_64-windows-msvc -S | FileCheck %s

define void @Vanilla() {
entry:
  ret void
}

define linkonce void @LinkOnce() {
entry:
  ret void
}

define weak void @Weak() {
entry:
  ret void
}

declare extern_weak void @ExternWeak()

define linkonce_odr void @LinkOnceOdr() {
entry:
  ret void
}

define weak_odr void @WeakOdr() {
entry:
  ret void
}

; CHECK: define void @Vanilla() comdat {
; CHECK: define linkonce void @LinkOnce() {
; CHECK: define weak void @Weak() {
; CHECK: declare extern_weak void @ExternWeak()
; CHECK: define linkonce_odr void @LinkOnceOdr() comdat {
; CHECK: define weak_odr void @WeakOdr() comdat {
