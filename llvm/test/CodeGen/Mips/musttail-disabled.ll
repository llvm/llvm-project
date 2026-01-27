; RUN: not --crash llc -mtriple=mips-unknown-linux-gnu < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=mips-unknown-linux-gnu -mips-tail-calls=0 < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=mips64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; Test that musttail fails when MIPS tail calls are disabled (default).

; CHECK: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail

define hidden i32 @callee(i32 %a) {
  ret i32 %a
}

define i32 @caller(i32 %a) {
  %ret = musttail call i32 @callee(i32 %a)
  ret i32 %ret
}
