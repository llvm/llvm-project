; RUN: rm -rf %t && split-file %s %t
; RUN: not llvm-as %t/denormal_fpenv_no_parens.ll -disable-output 2>&1 | FileCheck -check-prefix=NOPARENS %s
; RUN: not llvm-as %t/denormal_fpenv_lparen.ll -disable-output 2>&1 | FileCheck -check-prefix=LPAREN %s
; RUN: not llvm-as %t/denormal_fpenv_rparen.ll -disable-output 2>&1 | FileCheck -check-prefix=RPAREN %s
; RUN: not llvm-as %t/denormal_fpenv_empty_parens.ll -disable-output 2>&1 | FileCheck -check-prefix=EMPTYPARENS %s
; RUN: not llvm-as %t/invalid_single_entry.ll -disable-output 2>&1 | FileCheck -check-prefix=INVALID_SINGLE_ENTRY %s
; RUN: not llvm-as %t/invalid_multi_entry0.ll -disable-output 2>&1 | FileCheck -check-prefix=INVALID_MULTI_ENTRY0 %s
; RUN: not llvm-as %t/invalid_multi_entry1.ll -disable-output 2>&1 | FileCheck -check-prefix=INVALID_MULTI_ENTRY1 %s
; RUN: not llvm-as %t/invalid_second_element.ll -disable-output 2>&1 | FileCheck -check-prefix=INVALID_SECOND_ELEMENT %s
; RUN: not llvm-as %t/missing_bar.ll -disable-output 2>&1 | FileCheck -check-prefix=MISSING_BAR %s
; RUN: not llvm-as %t/with_comma.ll -disable-output 2>&1 | FileCheck -check-prefix=WITH_COMMA %s
; RUN: not llvm-as %t/missing_rparen_one_elt.ll -disable-output 2>&1 | FileCheck -check-prefix=MISSING_RPAREN_ONEELT %s
; RUN: not llvm-as %t/missing_rparen_two_elt.ll -disable-output 2>&1 | FileCheck -check-prefix=MISSING_RPAREN_TWOELT %s
; RUN: not llvm-as %t/extra_elt.ll -disable-output 2>&1 | FileCheck -check-prefix=EXTRA_ELT %s
; RUN: not llvm-as %t/arg_attr.ll -disable-output 2>&1 | FileCheck -check-prefix=ARG_ATTR %s
; RUN: not llvm-as %t/ret_attr.ll -disable-output 2>&1 | FileCheck -check-prefix=RET_ATTR %s
; RUN: not llvm-as %t/start_not_float_type.ll -disable-output 2>&1 | FileCheck -check-prefix=START_NOT_FLOAT %s
; RUN: not llvm-as %t/only_float.ll -disable-output 2>&1 | FileCheck -check-prefix=ONLY_FLOAT %s
; RUN: not llvm-as %t/only_float_colon.ll -disable-output 2>&1 | FileCheck -check-prefix=ONLY_FLOAT_COLON %s
; RUN: not llvm-as %t/only_float_mode_invalid_single_entry.ll -disable-output 2>&1 | FileCheck -check-prefix=FLOAT_INVALID_SINGLE_ELT %s
; RUN: not llvm-as %t/only_float_mode_invalid_two_entries.ll -disable-output 2>&1 | FileCheck -check-prefix=FLOAT_INVALID_TWO_ELT %s
; RUN: not llvm-as %t/only_float_mode_invalid_second_entry.ll -disable-output 2>&1 | FileCheck -check-prefix=FLOAT_INVALID_SECOND_ENTRY %s
; RUN: not llvm-as %t/both_sections_wrong_type.ll -disable-output 2>&1 | FileCheck -check-prefix=BOTH_SECTIONS_WRONG_TYPE %s
; RUN: not llvm-as %t/both_sections_invalid_float_entry0.ll -disable-output 2>&1 | FileCheck -check-prefix=BOTH_SECTIONS_INVALID_FLOAT_ENTRY0 %s
; RUN: not llvm-as %t/both_sections_invalid_float_entry1.ll -disable-output 2>&1 | FileCheck -check-prefix=BOTH_SECTIONS_INVALID_FLOAT_ENTRY1 %s
; RUN: not llvm-as %t/missing_comma_float.ll -disable-output 2>&1 | FileCheck -check-prefix=MISSING_COMMA_FLOAT %s


;--- denormal_fpenv_no_parens.ll

; NOPARENS: :36: error: expected '('
define void @func() denormal_fpenv {
  ret void
}

;--- denormal_fpenv_lparen.ll
; LPAREN: 37: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv( {
  ret void
}

;--- denormal_fpenv_rparen.ll

; RPAREN: :35: error: expected '('
define void @func() denormal_fpenv) {
  ret void
}

;--- denormal_fpenv_empty_parens.ll

; EMPTYPARENS: :36: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv() {
  ret void
}

;--- invalid_single_entry.ll

; INVALID_SINGLE_ENTRY: :36: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(invalid) {
  ret void
}

;--- invalid_multi_entry0.ll

; INVALID_MULTI_ENTRY0: :36: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(invalid|invalid) {
  ret void
}

;--- invalid_multi_entry1.ll

; INVALID_MULTI_ENTRY1: :36: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(invalid0|invalid1) {
  ret void
}

;--- invalid_second_element.ll

; INVALID_SECOND_ELEMENT: :44: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(dynamic|invalid1) {
  ret void
}

;--- missing_bar.ll

; MISSING_BAR: :49: error: unterminated denormal_fpenv
define void @func() denormal_fpenv(preservesign preservesign) {
  ret void
}

;--- with_comma.ll

; WITH_COMMA: :49: error: unterminated denormal_fpenv
define void @func() denormal_fpenv(preservesign,preservesign) {
  ret void
}

;--- missing_rparen_one_elt.ll

; MISSING_RPAREN_ONEELT: :49: error: unterminated denormal_fpenv
define void @func() denormal_fpenv(preservesign {
  ret void
}

;--- missing_rparen_two_elt.ll

; MISSING_RPAREN_TWOELT: :58: error: unterminated denormal_fpenv
define void @func() denormal_fpenv(preservesign| dynamic {
  ret void
}

;--- extra_elt.ll

; EXTRA_ELT: :53: error: unterminated denormal_fpenv
define void @func() denormal_fpenv(preservesign|ieee|preservesign) {
  ret void
}

;--- arg_attr.ll

; ARG_ATTR: :25: error: this attribute does not apply to parameters
define void @func(float denormal_fpenv(preservesign) %arg) {
  ret void
}

;--- ret_attr.ll

; RET_ATTR: :8: error: this attribute does not apply to return values
define denormal_fpenv(preservesign) float @func() {
  ret void
}

;--- start_not_float_type.ll

; START_NOT_FLOAT: :42: error: expected float:
define void @func() denormal_fpenv(double) {
  ret void
}

;--- only_float.ll

; ONLY_FLOAT: :41: error: expected ':' before float denormal_fpenv
define void @func() denormal_fpenv(float) {
  ret void
}

;--- only_float_colon.ll

; ONLY_FLOAT_COLON: :42: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(float:) {
  ret void
}

;--- only_float_mode_invalid_single_entry.ll

; FLOAT_INVALID_SINGLE_ELT: :42: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(float:invalid) {
  ret void
}

;--- only_float_mode_invalid_two_entries.ll

; FLOAT_INVALID_TWO_ELT: :42: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(float:invalid|invalid) {
  ret void
}

;--- only_float_mode_invalid_second_entry.ll

; FLOAT_INVALID_SECOND_ENTRY: :55: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(float:preservesign|invalid) {
  ret void
}

;--- both_sections_wrong_type.ll

; BOTH_SECTIONS_WRONG_TYPE: :61: error: expected float:
define void @func() denormal_fpenv(preservesign|ieee, double:dynamic|preservesign) {
  ret void
}

;--- both_sections_invalid_float_entry0.ll

; BOTH_SECTIONS_INVALID_FLOAT_ENTRY0: :61: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(preservesign|ieee, float:invalid|dynamic) {
  ret void
}

;--- both_sections_invalid_float_entry1.ll

; BOTH_SECTIONS_INVALID_FLOAT_ENTRY1: :69: error: expected denormal behavior kind (ieee, preservesign, positivezero, dynamic)
define void @func() denormal_fpenv(preservesign|ieee, float:dynamic|invalid) {
  ret void
}

;--- missing_comma_float.ll

; MISSING_COMMA_FLOAT: :54: error: expected ',' before float:
define void @func() denormal_fpenv(preservesign|ieee float:dynamic|invalid) {
  ret void
}
