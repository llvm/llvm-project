; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @f0() allockind("free")
declare void @f1() allockind("alloc,aligned,uninitialized")
declare void @f2() allockind("realloc,zeroed,aligned")
declare void @f3() allockind("fjord")
; CHECK: :[[#@LINE-1]]:30: error: unknown allockind fjord
