; RUN: not --crash llc -mtriple=sparc < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=sparcv9 < %s 2>&1 | FileCheck %s

; A musttail call with stack arguments must fail, not become a normal call.

; CHECK: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail

declare i32 @callee(i32, i32, i32, i32, i32, i32, i32)

define i32 @caller(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g) {
  %r = musttail call i32 @callee(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g)
  ret i32 %r
}
