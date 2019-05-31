; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'byval' type does not match parameter!
declare void @foo(i32* byval(i8))
