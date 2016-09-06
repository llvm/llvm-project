; RUN: opt -S -o - -functionattrs %s | FileCheck %s

; CHECK-NOT: readnone
declare void @llvm.assume(i1)
