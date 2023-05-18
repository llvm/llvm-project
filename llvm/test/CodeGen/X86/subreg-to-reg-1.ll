; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

; CHECK:     {{leal	.*[)], %e.*}}
; CHECK-NOT: {{leal	.*[)], %e.*}}

; Don't eliminate or coalesce away the explicit zero-extension!
; This is currently using an leal because of a 3-addressification detail,
; though this isn't necessary; The point of this test is to make sure
; a 32-bit add is used.

define i64 @foo(i64 %a) nounwind {
  %b = add i64 %a, 4294967295
  %c = and i64 %b, 4294967295
  %d = add i64 %c, 1
  ret i64 %d
}
