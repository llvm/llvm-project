; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: error: redefinition of global '@bar'

@foo = global i32 0
@bar = alias i32, ptr @foo
@bar = alias i32, ptr @foo
