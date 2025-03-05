; Test that every boring node is removed and all interesting distinct nodes remain after aggressive distinct metadata reduction.

; RUN: llvm-reduce --aggressive-named-md-reduction --test %python --test-arg %p/Inputs/reduce-distinct-metadata.py %s -o %t
; RUN: FileCheck %s < %t

; CHECK-NOT: {{.*}}boring{{.*}}

define void @main() {
  ret void
}

!named.metadata = !{!0, !2}
!llvm.test.other.metadata = !{}

!0 = distinct !{!"interesting_0", !1, !3, !4, !10, !11}
!1 = distinct !{!"interesting_1", !5, !7, !"something"}
!2 = distinct !{!"boring_0", !3, !4, i32 5}
!3 = distinct !{!"interesting_1", !3, !4}
!4 = distinct !{!"interesting_1", !6, i2 1}
!5 = distinct !{!"interesting_2", !8}
!6 = distinct !{!"interesting_2", !10}
!7 = distinct !{!"interesting_2", !12}
!8 = distinct !{!"interesting_3", !10, !9}
!9 = distinct !{!"interesting_3", !11, !13}
!10 = distinct !{!"boring_1", i32 50}
!11 = distinct !{!"boring_1", i32 2}
!12 = distinct !{!"boring_3", i2 1}
!13 = distinct !{!"interesting_4"}
