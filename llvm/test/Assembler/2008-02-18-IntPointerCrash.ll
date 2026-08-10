; RUN: not llvm-as %s 2>&1 | FileCheck %s
; PR2060

; CHECK: integer/byte constant must have integer/byte type

define ptr @foo() {
       ret ptr 0
}
