; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @f () {
1:
; CHECK: error: label expected to be numbered '0'
  ret void
}
