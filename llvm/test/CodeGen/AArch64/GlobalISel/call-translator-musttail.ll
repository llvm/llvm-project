; RUN: not llc %s -mtriple aarch64-apple-darwin -debug-only=aarch64-call-lowering -global-isel -global-isel-abort=2 -o - 2>&1 | FileCheck %s
; REQUIRES: asserts

; Verify that we fall back to SelectionDAG, and error out when we can't tail call musttail functions
; CHECK: ... Cannot tail call externally-defined function with weak linkage for this OS.
; CHECK-NEXT: Failed to lower musttail call as tail call
; CHECK-NEXT: warning: Instruction selection used fallback path for caller_weak
; CHECK-NEXT: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
declare extern_weak void @callee_weak()
define void @caller_weak() {
  musttail call void @callee_weak()
  ret void
}
