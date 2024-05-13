;; Check that we can parse RUN lines that have shell redirections
; RUN: opt %s -passes=instsimplify -S -o - 2>/dev/null | FileCheck %s --check-prefix=I32
; RUN: opt < %s -passes=instsimplify -S 2>&1 | FileCheck %s --check-prefix=I32
; RUN: sed 's/i32/i64/g' %s | opt -passes=instsimplify -S | FileCheck %s --check-prefix=I64

define i32 @common_sub_operand(i32 %X, i32 %Y) {
  %Z = sub i32 %X, %Y
  %Q = add i32 %Z, %Y
  ret i32 %Q
}
