; RUN: llvm-extract -func foo -S %s | FileCheck %s

; llvm-extract should not copy ifuncs into the new module, so make sure they
; are turned into declarations.

; CHECK: define void @foo() {
; CHECK: call void @ifunc()
define void @foo() {
  call void @ifunc()
  ret void
}

define void @ifunc_impl() {
  ret void
}

; CHECK: declare void @ifunc()
@ifunc = ifunc void (), ptr @ifunc_resolver

define internal ptr @ifunc_resolver() {
  ret ptr @ifunc_impl
}
