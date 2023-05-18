; RUN: llvm-link -S %s %S/Inputs/associated-global.ll | FileCheck %s

; CHECK: @c = internal global i32 1, !associated !0
; CHECK: @d = global i32 0, !associated !1
; CHECK: @a = global i32 0, !associated !2
; CHECK: @b = global i32 0, !associated !3
; CHECK: @c.3 = internal global i32 1, !associated !4
; CHECK: @e = global i32 0, !associated !5

; CHECK: !0 = !{ptr @d}
; CHECK: !1 = !{ptr @c}
; CHECK: !2 = !{ptr @b}
; CHECK: !3 = !{ptr @a}
; CHECK: !4 = !{ptr @e}
; CHECK: !5 = !{ptr @c.3}


@a = external global i32, !associated !0
@b = global i32 0, !associated !1
@c = internal global i32 1, !associated !2
@d = global i32 0, !associated !3

!0 = !{ptr @b}
!1 = !{ptr @a}
!2 = !{ptr @d}
!3 = !{ptr @c}
