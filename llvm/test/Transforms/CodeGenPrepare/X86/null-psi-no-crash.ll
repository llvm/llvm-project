; RUN: not opt -passes=codegenprepare -mtriple=x86_64-linux-gnu -disable-output < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: this pass requires the profile-summary module analysis to be available
define void @f() {
  ret void
}
