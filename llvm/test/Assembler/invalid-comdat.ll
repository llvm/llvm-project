; RUN: not llvm-as < %s 2>&1 | FileCheck %s

@v = global i32 0, comdat($v)
; CHECK: use of undefined comdat '$v'
