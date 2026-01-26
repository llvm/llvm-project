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
  ;; The absolute_symbol range fits in 32 bits but we still use movabs
  ;; since there's no benefit to using the sign extending instruction
  ;; with absolute symbols.
  ; CHECK: movabsq $abs_sym_in_range, %rax
  %1 = ptrtoint ptr @abs_sym_in_range to i64
  ret i64 %1
}

@no_abs_sym = external hidden global [0 x i8]
@abs_sym = external hidden global [0 x i8], !absolute_symbol !0
@abs_sym_in_range = external hidden global [0 x i8], !absolute_symbol !1

!0 = !{i64 -1, i64 -1}  ;; Full range
!1 = !{i64 -2147483648, i64 2147483648}  ;; In range
