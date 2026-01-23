; RUN: llvm-as < %s | llvm-dis - | FileCheck %s

; CHECK: define void @f() section "foo_section"
; CHECK-NOT: "implicit-section-name"

define void @f() "implicit-section-name"="foo_section" {
  ret void
}
