; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: the range represent the empty set but limits aren't 0!
define void @range_empty(i8 range(i8 1, 1) %a) {
  ret void
}
