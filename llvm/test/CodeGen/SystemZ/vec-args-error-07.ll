; Verify that we detect unsupported single-element vector types.

; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 2>&1 | FileCheck %s

declare void @bar(<1 x fp128>)

define void @foo() {
  call void @bar (<1 x fp128> <fp128 0xL00000000000000000000000000000000>)
  ret void
}

; CHECK: LLVM ERROR: Unsupported vector argument or return type
