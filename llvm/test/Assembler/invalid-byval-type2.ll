; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: void type only allowed for function results
declare void @foo(ptr byval(void))
