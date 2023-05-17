; RUN: llc -O0 -fast-isel -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

; CHECK-NOT: retq
; CHECK: jmpq

define void @f(ptr %this) "disable-tail-calls"="true" {
  musttail call void %this(ptr %this)
  ret void
}
