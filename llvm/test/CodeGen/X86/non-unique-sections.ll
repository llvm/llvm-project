; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -unique-section-names=false | FileCheck %s

; CHECK:   .section                      .text,"ax",@progbits,unique
; CHECK-NOT: section
; CHECK: f:
define void @f() {
  ret void
}

; CHECK:   .section                      .text,"ax",@progbits,unique
; CHECK-NOT: section
; CHECK: g:
define void @g() {
  ret void
}
