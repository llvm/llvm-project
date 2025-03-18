; RUN: llc -mtriple=hexagon -hexagon-expand-condsets=0 < %s | FileCheck %s

; In the IR, the i1 value is zero-extended first, then passed to add.
; Check that in the final code, the mux happens after the add.
; CHECK: [[REG1:r[0-9]+]] = add([[REG0:r[0-9]+]],#1)
; CHECK: r{{[0-9]+}} = mux(p{{[0-3]}},[[REG1]],[[REG0]])

define i32 @foo(i32 %a, i32 %b) {
  %v0 = icmp eq i32 %a, %b
  %v1 = zext i1 %v0 to i32
  %v2 = add i32 %v1, %a
  ret i32 %v2
}
