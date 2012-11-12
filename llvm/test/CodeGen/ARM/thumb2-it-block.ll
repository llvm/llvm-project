; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s
; PR11107

define i32 @test(i32 %a, i32 %b) {
entry:
; CHECK:        cmp
; CHECK-NEXT:   it    mi
; CHECK-NEXT:   rsbmi
; CHECK-NEXT:   cmp
; CHECK-NEXT:   it    mi
; CHECK-NEXT:   rsbmi
 %cmp1 = icmp slt i32 %a, 0
 %sub1 = sub nsw i32 0, %a
 %abs1 = select i1 %cmp1, i32 %sub1, i32 %a
 %cmp2 = icmp slt i32 %b, 0
 %sub2 = sub nsw i32 0, %b
 %abs2 = select i1 %cmp2, i32 %sub2, i32 %b
 %add = add nsw i32 %abs1, %abs2
 ret i32 %add
}
