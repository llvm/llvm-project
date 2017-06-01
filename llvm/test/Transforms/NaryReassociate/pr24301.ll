; RUN: opt < %s -nary-reassociate -S | FileCheck %s
; RUN: opt < %s -passes='nary-reassociate' -S | FileCheck %s

define i32 @foo(i32 %tmp4) {
; CHECK-LABEL: @foo(
entry:
  %tmp5 = add i32 %tmp4, 8
  %tmp13 = add i32 %tmp4, -128  ; deleted
  %tmp14 = add i32 %tmp13, 8    ; => %tmp5 + -128
  %tmp21 = add i32 119, %tmp4
  ; do not rewrite %tmp23 against %tmp13 because %tmp13 is already deleted
  %tmp23 = add i32 %tmp21, -128
; CHECK: %tmp23 = add i32 %tmp21, -128
  ret i32 %tmp23
}
