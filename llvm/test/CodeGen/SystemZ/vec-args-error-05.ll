; Verify that we detect unsupported single-element vector types.

; RUN: not --crash llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 2>&1 | FileCheck %s

define void @foo(<1 x fp128>) {
  ret void
}

; CHECK: LLVM ERROR: Unsupported vector argument or return type
