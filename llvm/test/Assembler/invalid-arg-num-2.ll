; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: argument expected to be numbered '%1'
define void @foo(i8 %0, i32 %named, i32 %2) {
  ret void
}
