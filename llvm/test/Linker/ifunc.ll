; RUN: split-file %s %t
; RUN: llvm-link %t/a.ll %t/b.ll -S -o - | FileCheck %s

;; Check that ifuncs are linked in properly.

; CHECK-DAG: @foo = ifunc void (), void ()* ()* @foo_resolve
; CHECK-DAG: define internal void ()* @foo_resolve() {

; CHECK-DAG: @bar = ifunc void (), void ()* ()* @bar_resolve
; CHECK-DAG: define internal void ()* @bar_resolve() {

;--- a.ll
declare void @bar()

;--- b.ll
@foo = ifunc void (), void ()* ()* @foo_resolve
@bar = ifunc void (), void ()* ()* @bar_resolve

define internal void ()* @foo_resolve() {
  ret void ()* null
}

define internal void ()* @bar_resolve() {
  ret void ()* null
}
