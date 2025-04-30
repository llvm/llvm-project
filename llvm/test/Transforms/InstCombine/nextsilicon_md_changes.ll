; RUN: opt %s -passes=instcombine -S | FileCheck %s

define i32 @ne(i32 %X, i32 %Y) {
; CHECK-LABEL: @ne(
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT:    br i1 [[C]], label [[F:%.*]], label [[T:%.*]], !prof ![[PROF_MD:[0-9]+]], !nextsilicon ![[NS_MD:[0-9]+]]
; CHECK:       T:
; CHECK-NEXT:    ret i32 12
; CHECK:       F:
; CHECK-NEXT:    ret i32 123
;
  %C = icmp ne i32 %X, %Y
  br i1 %C, label %T, label %F, !prof !0, !nextsilicon !1
T:
  ret i32 12
F:
  ret i32 123
}

!0 = !{!"branch_weights", i32 1,  i32 99}
!1 = !{!"branch_counts", i32 2,  i32 99}

; CHECK: ![[PROF_MD]] = {{.*}} i32 99, i32 1}
; CHECK: ![[NS_MD]] = {{.*}} i32 99, i32 2}
