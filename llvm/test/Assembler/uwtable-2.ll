; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

declare void @f() uwtable(sync x
; CHECK: :[[#@LINE-1]]:32: error: expected ')'
