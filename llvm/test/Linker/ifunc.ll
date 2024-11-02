; RUN: split-file %s %t
; RUN: llvm-link %t/a.ll %t/b.ll -S -o - | FileCheck %s

;; Check that ifuncs are linked in properly.

; CHECK-DAG: @foo = ifunc void (), ptr @foo_resolve
; CHECK-DAG: define internal ptr @foo_resolve() {

; CHECK-DAG: @bar = ifunc void (), ptr @bar_resolve
; CHECK-DAG: define internal ptr @bar_resolve() {

;--- a.ll
declare void @bar()

;--- b.ll
@foo = ifunc void (), ptr @foo_resolve
@bar = ifunc void (), ptr @bar_resolve

define internal ptr @foo_resolve() {
  ret ptr null
}

define internal ptr @bar_resolve() {
  ret ptr null
}
