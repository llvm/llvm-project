; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; CHECK: %t = add i64 %t, 1
; CHECK: -->  %t

define void @foo() {
entry:
  ret void

dead:
  %t = add i64 %t, 1
  ret void
}
