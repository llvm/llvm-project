; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; CHECK: %t = add i64 %t, 1
; CHECK: -->  undef

define void @foo() {
entry:
  ret void

dead:
  %t = add i64 %t, 1
  ret void
}
