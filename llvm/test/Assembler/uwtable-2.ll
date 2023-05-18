; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @f() uwtable(sync x
; CHECK: :[[#@LINE-1]]:32: error: expected ')'
