; RUN: split-file %s %t
; RUN: not llvm-as < %s %t/outer_left_parenthesis.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OUTER-LEFT
; RUN: not llvm-as < %s %t/inner_left_parenthesis.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INNER-LEFT
; RUN: not llvm-as < %s %t/inner_right_parenthesis.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INNER-RIGHT
; RUN: not llvm-as < %s %t/outer_right_parenthesis.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OUTER-RIGHT
; RUN: not llvm-as < %s %t/integer.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INTEGER
; RUN: not llvm-as < %s %t/lower_equal_upper.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOWER-EQUAL-UPPER
; RUN: not llvm-as < %s %t/inner_comma.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INNER-COMMA
; RUN: not llvm-as < %s %t/outer_comma.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OUTER-COMMA
; RUN: not llvm-as < %s %t/empty1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMPTY1
; RUN: not llvm-as < %s %t/empty2.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMPTY2

;--- outer_left_parenthesis.ll
; OUTER-LEFT: expected '('
define void @foo(ptr initializes 0, 4 %a) {
  ret void
}

;--- inner_left_parenthesis.ll
; INNER-LEFT: expected '('
define void @foo(ptr initializes(0, 4 %a) {
  ret void
}

;--- inner_right_parenthesis.ll
; INNER-RIGHT: expected ')'
define void @foo(ptr initializes((0, 4 %a) {
  ret void
}

;--- outer_right_parenthesis.ll
; OUTER-RIGHT: expected ')'
define void @foo(ptr initializes((0, 4) %a) {
  ret void
}

;--- integer.ll
; INTEGER: expected integer
define void @foo(ptr initializes((0.5, 4)) %a) {
  ret void
}

;--- lower_equal_upper.ll
; LOWER-EQUAL-UPPER: the range should not represent the full or empty set!
define void @foo(ptr initializes((4, 4)) %a) {
  ret void
}

;--- inner_comma.ll
; INNER-COMMA: expected ','
define void @foo(ptr initializes((0 4)) %a) {
  ret void
}

;--- outer_comma.ll
; OUTER-COMMA: expected ')'
define void @foo(ptr initializes((0, 4) (8, 12)) %a) {
  ret void
}

;--- empty1.ll
; EMPTY1: expected '('
define void @foo(ptr initializes() %a) {
  ret void
}

;--- empty2.ll
; EMPTY2: expected integer
define void @foo(ptr initializes(()) %a) {
  ret void
}
