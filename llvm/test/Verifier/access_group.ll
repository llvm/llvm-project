; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

define void @test(ptr %p) {
; CHECK: Access scope list contains invalid access scope
  load i8, ptr %p, !llvm.access.group !1
; CHECK: Access scope list must consist of MDNodes
  load i8, ptr %p, !llvm.access.group !2
; CHECK-NOT: Access scope
  load i8, ptr %p, !llvm.access.group !3
  load i8, ptr %p, !llvm.access.group !4
  ret void
}

!0 = !{}
!1 = !{!0}
!2 = !{!"foo"}
!3 = distinct !{}
!4 = !{!3}
