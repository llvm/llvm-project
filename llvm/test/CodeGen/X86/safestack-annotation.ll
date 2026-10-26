; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; The unsafe stack size is passed in the "unsafe-stack-size" function
; attribute. An annotation with the same name is not read by codegen.

; CHECK-LABEL: f:
define void @f() safestack !annotation !0 {
  ret void
}

!0 = !{!"unsafe-stack-size", !"not-an-integer"}
