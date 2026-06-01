; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!cfi.functions = !{!0, !2, !4}
!0 = !{!"function", i8 0, !1}
!1 = !{i64 0, !"typeid1"}
!2 = !{!"other_function", i8 1, !1, !3}
!3 = !{i64 0, !"typeid2"}
!4 = !{!"up_to_date", i8 2, i64 123456789, !1}

; CHECK: !cfi.functions = !{!0, !2, !4}
; CHECK: !0 = !{!"function", i8 0, i64 6717233803957748929, !1}
; CHECK: !1 = !{i64 0, !"typeid1"}
; CHECK: !2 = !{!"other_function", i8 1, i64 -2568568921219972102, !1, !3}
; CHECK: !3 = !{i64 0, !"typeid2"}
; CHECK: !4 = !{!"up_to_date", i8 2, i64 123456789, !1}
