; Verify that we detect unsupported single-element vector types.

; RUN: not llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 2>&1 | FileCheck %s

define <1 x fp128> @foo() {
  ret <1 x fp128><fp128 0xL00000000000000000000000000000000>
}

; CHECK: LLVM ERROR: Unsupported vector argument or return type
