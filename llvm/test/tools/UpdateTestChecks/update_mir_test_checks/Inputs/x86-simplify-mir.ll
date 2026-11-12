; RUN: llc < %s -mtriple=x86_64 -stop-after=finalize-isel -simplify-mir | FileCheck %s

define void @f() {
  ret void
}
