; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attributes 'byval'{{.*}} do not support unsized types!
declare void @foo(void()* byval(void()))
