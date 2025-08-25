; RUN: not llvm-as %s 2>&1 | FileCheck %s
; PR2060

; CHECK: integer constant must have integer type

define ptr @foo() {
       ret ptr 0
}
