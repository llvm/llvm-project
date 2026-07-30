@a = global i32 0, !associated !0
@b = external global i32, !associated !1
@c = internal global i32 1, !associated !2
@e = global i32 0, !associated !3

!0 = !{ptr @b}
!1 = !{ptr @a}
!2 = !{ptr @e}
!3 = !{ptr @c}
