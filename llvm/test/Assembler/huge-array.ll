; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: define ptr @foo() {
; CHECK: ret ptr null
define ptr @foo() {
  ret ptr null
}
