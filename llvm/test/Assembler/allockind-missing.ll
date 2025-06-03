; RUN: not llvm-as --disable-output %s  2>&1 | FileCheck -DFILE=%s %s

; CHECK: [[FILE]]:[[@LINE+1]]:30: error: expected allockind value
declare void @f0() allockind()
