; RUN: opt -S -o - -passes=function-attrs %s | FileCheck %s

; CHECK-NOT: readnone
declare void @llvm.assume(i1)
