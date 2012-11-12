;RUN: opt -S -std-link-opts < %s | FileCheck %s
; Simple test to check that -std-link-opts keeps only the main function.

; CHECK-NOT: define
; CHECK: define void @main
; CHECK-NOT: define
define void @main() {
  ret void
}

define void @foo() {
  ret void
}
