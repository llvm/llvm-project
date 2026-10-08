target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vt2a = constant ptr @vf2a, !type !0, !guid !1
@vt2b = constant ptr @vf2b, !type !0, !guid !2

define internal i1 @vf2a(ptr %this) !guid !3 {
  ret i1 0
}

define internal i1 @vf2b(ptr %this) !guid !4 {
  ret i1 1
}

!0 = !{i32 0, !"typeid2"}
!1 = !{i64 327052092150035973}
!2 = !{i64 -3123971987909853701}
!3 = !{i64 520420412982578150}
!4 = !{i64 6199549476356213762}
