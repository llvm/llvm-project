; RUN: llvm-diff %s %s 2>&1 | FileCheck %s

; CHECK: not comparing 1 anonymous functions in the left module and 1 in the right module

define void @0() {
  ret void
}

