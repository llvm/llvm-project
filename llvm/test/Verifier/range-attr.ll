; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Range bit width must match type bit width!
define void @bit_widths_do_not_match(i32 range(i8,1,0) %a) {
  ret void
}

; CHECK: Range bit width must match type bit width!
define void @bit_widths_do_not_match_vector(<4 x i32> range(i8,1,0) %a) {
  ret void
}
