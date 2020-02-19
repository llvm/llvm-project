; Verify that we detect unsupported single-element vector types.

; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 2>&1 | FileCheck %s

declare void @bar(<1 x i128>)

define void @foo() {
  call void @bar (<1 x i128> <i128 0>)
  ret void
}

; CHECK: LLVM ERROR: Unsupported vector argument or return type
