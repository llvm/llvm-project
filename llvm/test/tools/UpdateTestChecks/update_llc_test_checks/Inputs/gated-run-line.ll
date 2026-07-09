; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefixes=PLAIN
; RUN(some-feature): llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s --check-prefixes=GATED

define i32 @add(i32 %a, i32 %b) {
entry:
  %r = add i32 %a, %b
  ret i32 %r
}
