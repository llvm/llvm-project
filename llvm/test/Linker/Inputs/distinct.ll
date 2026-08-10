@global = linkonce global i32 0

!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = !{}
!1 = !{!0}
!2 = !{ptr @global}
!3 = distinct !{}
!4 = distinct !{!0}
!5 = distinct !{ptr @global}
!6 = !{!3}
!7 = !{!4}
!8 = !{!5}
