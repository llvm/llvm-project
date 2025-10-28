; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
; CHECK-NOT: memd
; CHECK: call f1
; CHECK: r{{[0-9]}}:{{[0-9]}} = combine(#0,#10)
target triple = "hexagon"

define i64 @f0(i32 %a0) {
b0:
  %v0 = add nsw i32 %a0, 5
  %v1 = tail call i64 @f1(i32 %v0)
  %v2 = add nsw i64 %v1, 10
  ret i64 %v2
}

declare i64 @f1(i32)
