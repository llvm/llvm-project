; RUN: llc --code-model=kernel < %s -asm-verbose=0 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: func_abs_sym
define i64 @func_abs_sym() nounwind {
  ; CHECK: movabsq $abs_sym, %rax
  %1 = ptrtoint ptr @abs_sym to i64
  ret i64 %1
}

; CHECK-LABEL: func_no_abs_sym
define i64 @func_no_abs_sym() nounwind {
  ; CHECK: movq $no_abs_sym, %rax
  %1 = ptrtoint ptr @no_abs_sym to i64
  ret i64 %1
}

@abs_sym = external hidden global [0 x i8], !absolute_symbol !0
@no_abs_sym = external hidden global [0 x i8]

!0 = !{i64 -1, i64 -1}
