; RUN: opt < %s -gvn -S | FileCheck %s
; This testcase tests insertion of no-cost phis.  That is,
; when the value is already available in every predecessor,
; and we just need to insert a phi node to merge the available values.

@c = global i32 0, align 4
@d = global i32 0, align 4


define i32 @mai(i32 %foo, i32 %a, i32 %b) {
  %1 = icmp ne i32 %foo, 0
  br i1 %1, label %bb1, label %bb2

bb1:
  %2 = add nsw i32 %a, %b
  store i32 %2, i32* @c, align 4
  br label %mergeblock

bb2:
  %3 = add nsw i32 %a, %b
  store i32 %3, i32* @d, align 4
  br label %mergeblock

mergeblock:
; CHECK: pre-phi = phi i32 [ %3, %bb2 ], [ %2, %bb1 ]
; CHECK-NEXT: ret i32 %.pre-phi
  %4 = add nsw i32 %a, %b
  ret i32 %4
}


