; RUN: llc -O0 -march=amdgcn < %s | FileCheck %s

; CHECK-NOT: foo
define internal i32 @foo() {
  ret i32 0
}

; CHECK-NOT: bar
define private i32 @bar() {
  ret i32 0
}
