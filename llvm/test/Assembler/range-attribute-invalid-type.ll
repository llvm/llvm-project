; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: the range must have integer type!
define void @range_vector_type(i8 range(<4 x i32> 0, 0) %a) {
  ret void
}
