; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: argument expected to be numbered '%0'
define void @foo(i8 %1) {
  ret void
}
