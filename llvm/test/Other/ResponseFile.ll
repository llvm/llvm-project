; RUN: echo %s > %t.list
; RUN: llvm-as @%t.list -o %t.bc
; RUN: llvm-nm %t.bc 2>&1 | FileCheck %s

; CHECK: T foobar

define void @foobar() {
  ret void
}
