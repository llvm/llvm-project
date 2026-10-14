; RUN: not llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=-x87 2>&1 | FileCheck %s

; Verify that returning an x86_fp80 value with x87 disabled on x86_64 produces
; a clear, user-friendly diagnostic instead of crashing with
; "Cannot select: build_pair".
; See: https://github.com/llvm/llvm-project/issues/182450

define x86_fp80 @test_ret_f80(x86_fp80 %x) {
entry:
  ret x86_fp80 %x
}

; CHECK: LLVM ERROR: cannot use x86_fp80 type with x87 disabled on x86_64 target
