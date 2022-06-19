; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @f0() allockind()
; CHECK: :[[#@LINE-1]]:30: error: expected allockind value
