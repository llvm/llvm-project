; RUN: llc --code-model=kernel < %s -asm-verbose=0 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: func_no_abs_sym
define i64 @func_no_abs_sym() nounwind {
  ; CHECK: movq $no_abs_sym, %rax
  %1 = ptrtoint ptr @no_abs_sym to i64
  ret i64 %1
}

; CHECK-LABEL: func_abs_sym
define i64 @func_abs_sym() nounwind {
  ; CHECK: movabsq $abs_sym, %rax
  %1 = ptrtoint ptr @abs_sym to i64
  ret i64 %1
}

; CHECK-LABEL: func_abs_sym_in_range
define i64 @func_abs_sym_in_range() nounwind {
  ; CHECK: movq $abs_sym_in_range, %rax
  %1 = ptrtoint ptr @abs_sym_in_range to i64
  ret i64 %1
}

; CHECK-LABEL: func_abs_sym_min_out_of_range
define i64 @func_abs_sym_min_out_of_range() nounwind {
  ; CHECK: movabsq $abs_sym_min_out_of_range, %rax
  %1 = ptrtoint ptr @abs_sym_min_out_of_range to i64
  ret i64 %1
}

; CHECK-LABEL: func_abs_sym_max_out_of_range
define i64 @func_abs_sym_max_out_of_range() nounwind {
  ; CHECK: movabsq $abs_sym_max_out_of_range, %rax
  %1 = ptrtoint ptr @abs_sym_max_out_of_range to i64
  ret i64 %1
}

@no_abs_sym = external hidden global [0 x i8]
@abs_sym = external hidden global [0 x i8], !absolute_symbol !0
@abs_sym_in_range = external hidden global [0 x i8], !absolute_symbol !1
@abs_sym_min_out_of_range = external hidden global [0 x i8], !absolute_symbol !2
@abs_sym_max_out_of_range = external hidden global [0 x i8], !absolute_symbol !3

!0 = !{i64 -1, i64 -1}  ;; Full range
!1 = !{i64 -2147483648, i64 2147483648}  ;; Note the upper bound is exclusive.
!2 = !{i64 -2147483649, i64 2147483648}  ;; Min is one below -2^31
!3 = !{i64 -2147483648, i64 2147483649}  ;; Max is one above 2^31-1
