; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: argument expected to be numbered '%1'
define void @foo(i32 %0, i32 %5) {
  ret void
}
