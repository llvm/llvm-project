; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

define i64 @foo(i64 %a) {
entry:
  %b = add i64 %a, 1
  ret i64 %b
}
