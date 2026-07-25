; RUN: llc -mtriple=aarch64-apple-macosx13.0.0 -filetype=obj %s -o %t.o
; RUN: llvm-nm -m %t.o | FileCheck %s

define internal void @foo_internal() {
  ret void
}

@foo_default = weak_odr alias void (), ptr @foo_internal
@foo_hidden = weak_odr hidden alias void (), ptr @foo_internal

define weak_odr hidden void @foo_defined() {
  ret void
}

; CHECK-DAG: weak external _foo_default
; CHECK-DAG: weak private external _foo_hidden
; CHECK-DAG: weak private external _foo_defined
