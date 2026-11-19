; RUN: not --crash llc -mtriple=aarch64-unknown-linux-gnu -o - < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=aarch64-unknown-fuchsia -o - < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Calling convention AArch64_Outline_Atomics_PreserveMost is only supported for calls to out-of-line atomic helper routines.
define aarch64_outline_atomics_preservemost void @bad_callee() {
  unreachable
}
