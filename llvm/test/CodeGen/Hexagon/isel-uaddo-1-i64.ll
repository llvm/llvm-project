; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: add{{.*}}:carry

target triple = "hexagon-unknown-linux-gnu"

define i64 @f0(i64 %a0, ptr %a1) {
b0:
  %v0 = add i64 -9223372036854775808, %a0
  %v1 = icmp ugt i64 -9223372036854775808, %v0
  store i1 %v1, ptr %a1, align 1
  ret i64 %v0
}
