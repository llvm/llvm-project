!llvm.module.flags = !{!3, !4, !5}

!0 = !{}
!1 = !{!0}
!2 = !{!0, !1}
!3 = !{i32 4, !"foo", i32 37} ; Override the "foo" value.
!4 = !{i32 5, !"bar", !1}
!5 = !{i32 6, !"baz", !2}
