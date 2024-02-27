; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; Check the error message on full range attribute.

; CHECK: The range should not represent the full or empty set!
define void @range_empty(i8 range(i8,0,0) %a) {
  ret void
}
