; RUN: not llc %s -mtriple aarch64-unknown-unknown -debug-only=aarch64-call-lowering -global-isel -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: Cannot lower musttail calls yet.
; CHECK-NEXT: LLVM ERROR: unable to translate instruction: call (in function: foo)
declare void @must_callee(i8*)
define void @foo(i32*) {
  musttail call void @must_callee(i8* null)
  ret void
}
