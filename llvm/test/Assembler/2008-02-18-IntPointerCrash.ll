; RUN: not llvm-as %s 2>&1 | grep "integer constant must have integer type"
; PR2060

define ptr @foo() {
       ret ptr 0
}
