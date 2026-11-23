; RUN: llc -mtriple=mips-unknown-linux-gnu -mips-tail-calls=1 -O0 < %s | FileCheck %s
; RUN: llc -mtriple=mips64-unknown-linux-gnu -mips-tail-calls=1 -O0 < %s | FileCheck %s

; Test that musttail works correctly at -O0 when Fast ISel is used.
; This is a regression test for a bug where Fast ISel incorrectly set
; IncomingArgSize to 0 for functions with no arguments, causing the
; tail call eligibility check to fail.

@ptr = dso_local global ptr null, align 4

define dso_local void @callee() {
entry:
  %local = alloca i32, align 4
  store volatile i32 2, ptr %local, align 4
  ret void
}

; CHECK-LABEL: caller:
; CHECK: j callee
; CHECK-NOT: jal callee
define dso_local void @caller() {
entry:
  %local = alloca i32, align 4
  store i32 1, ptr %local, align 4
  store ptr %local, ptr @ptr, align 4
  musttail call void @callee()
  ret void
}
