; RUN: llvm-as < %s -disable-output 2>&1 | FileCheck %s -allow-empty
; CHECK-NOT: error
; CHECK-NOT: warning
; RUN: verify-uselistorder < %s

define void @f1(i1 %cmp) {
entry:
  br i1 %cmp, label %branch, label %branch
branch:
  unreachable
}
