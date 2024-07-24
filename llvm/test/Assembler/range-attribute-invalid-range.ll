; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: the range represent the full or empty set but they aren't min or max value!
define void @range_empty(i8 range(i8 3, 3) %a) {
  ret void
}
