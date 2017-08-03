; RUN: llc < %s -fast-isel -fast-isel-abort=1 -mtriple=x86_64-apple-darwin10 | FileCheck %s
; RUN: llc < %s -fast-isel -march=x86 -mattr=+sse2 | FileCheck --check-prefix=SSE2 %s

; SSE2: xor
; SSE2: xor
; SSE2-NOT: xor

; CHECK-LABEL: doo:
; CHECK: xor
define double @doo(double %x) nounwind {
  %y = fsub double -0.0, %x
  ret double %y
}

; CHECK-LABEL: foo:
; CHECK: xor
define float @foo(float %x) nounwind {
  %y = fsub float -0.0, %x
  ret float %y
}
