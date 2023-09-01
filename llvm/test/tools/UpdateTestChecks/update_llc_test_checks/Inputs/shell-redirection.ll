;; Check that we can parse RUN lines that have shell redirections
; RUN: llc -mtriple=x86_64 %s -o - 2>/dev/null | FileCheck %s --check-prefix=I32
; RUN: llc < %s -mtriple=x86_64 2>&1 | FileCheck %s --check-prefix=I32
; RUN: sed 's/i32/i64/g' %s | llc -mtriple=x86_64 2>&1 | FileCheck %s --check-prefix=I64

define i32 @add(i32 %X, i32 %Y) {
  %Q = add i32 %X, %Y
  ret i32 %Q
}
