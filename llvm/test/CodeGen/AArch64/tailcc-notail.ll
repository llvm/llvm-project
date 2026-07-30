; RUN: not --crash llc -mtriple=arm64-apple-ios %s -o - 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: failed to perform tail call elimination on a call site marked musttail
declare tailcc [16 x i64] @callee()
define tailcc [16 x i64] @caller() {
  %res = musttail call tailcc [16 x i64] @callee()
  ret [16 x i64] %res
}
