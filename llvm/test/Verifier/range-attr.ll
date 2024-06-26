; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Range bit width must match type bit width!
; CHECK-NEXT: ptr @bit_widths_do_not_match
define void @bit_widths_do_not_match(i32 range(i8 1, 0) %a) {
  ret void
}

; CHECK: Range bit width must match type bit width!
; CHECK-NEXT: ptr @bit_widths_do_not_match_vector
define void @bit_widths_do_not_match_vector(<4 x i32> range(i8 1, 0) %a) {
  ret void
}

; CHECK: Attribute 'range(i8 1, 0)' applied to incompatible type!
; CHECK-NEXT: ptr @not-integer-type
define void @not-integer-type(ptr range(i8 1, 0) %a) {
  ret void
}
