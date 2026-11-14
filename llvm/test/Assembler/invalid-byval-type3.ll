; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: Attribute 'byval' does not support unsized types!
declare void @foo(ptr byval(void()))
