; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: error: void type only allowed for function results
declare void @foo(ptr byval(void))
