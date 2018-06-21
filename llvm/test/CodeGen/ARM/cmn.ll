; RUN: llc < %s -mtriple thumbv7-apple-ios | FileCheck %s
; <rdar://problem/7569620>

define i32 @compare_i_gt(i32 %a) {
entry:
; CHECK:     compare_i_gt
; CHECK-NOT: mvn
; CHECK:     cmn
  %cmp = icmp sgt i32 %a, -78
  %. = zext i1 %cmp to i32
  ret i32 %.
}

define i32 @compare_r_eq(i32 %a, i32 %b) {
entry:
; CHECK: compare_r_eq
; CHECK: cmn
  %sub = sub nsw i32 0, %b
  %cmp = icmp eq i32 %a, %sub
  %. = zext i1 %cmp to i32
  ret i32 %.
}
