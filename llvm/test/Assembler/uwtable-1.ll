; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @f0() uwtable
declare void @f1() uwtable(sync)
declare void @f2() uwtable(async)
declare void @f3() uwtable(unsync)
; CHECK: :[[#@LINE-1]]:28: error: expected unwind table kind
