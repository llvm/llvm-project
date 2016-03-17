; RUN: opt < %s -S -inline | FileCheck %s

; Make sure that profile and unpredictable  metadata is preserved when cloning a select.

define i32 @callee_with_select(i1 %c, i32 %a, i32 %b) {
  %sel = select i1 %c, i32 %a, i32 %b, !prof !0, !unpredictable !1
  ret i32 %sel
}

define i32 @caller_of_select(i1 %C, i32 %A, i32 %B) {
  %ret = call i32 @callee_with_select(i1 %C, i32 %A, i32 %B)
  ret i32 %ret

; CHECK-LABEL: @caller_of_select(
; CHECK-NEXT:  [[SEL:%.*]] = select i1 %C, i32 %A, i32 %B, !prof !0, !unpredictable !1
; CHECK-NEXT:  ret i32 [[SEL]]
}

; Make sure that profile and unpredictable metadata is preserved when cloning a branch.

define i32 @callee_with_branch(i1 %c) {
  br i1 %c, label %if, label %else, !unpredictable !1, !prof !2
if:
  ret i32 1
else:
  ret i32 2
}

define i32 @caller_of_branch(i1 %C) {
  %ret = call i32 @callee_with_branch(i1 %C)
  ret i32 %ret

; CHECK-LABEL: @caller_of_branch(
; CHECK-NEXT:  br i1 %C, label %{{.*}}, label %{{.*}}, !prof !2, !unpredictable !1
}

!0 = !{!"branch_weights", i32 1, i32 2}
!1 = !{}
!2 = !{!"branch_weights", i32 3, i32 4}

; CHECK: !0 = !{!"branch_weights", i32 1, i32 2}
; CHECK: !1 = !{}
; CHECK: !2 = !{!"branch_weights", i32 3, i32 4}

