; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

define i32 @dot_value_and_metadata_string(ptr %p, i32 %n) {
  %.sum = add i32 %n, 1
  store i32 %.sum, ptr %p, align 4, !alias.scope !0
  ret i32 %.sum
}

!0 = !{!1}
!1 = distinct !{!1, !2, !"callee: %.arg"}
!2 = distinct !{!2, !"callee"}
