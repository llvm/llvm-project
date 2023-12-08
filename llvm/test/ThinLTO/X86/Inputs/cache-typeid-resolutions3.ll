target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vt2a = constant ptr @vf2a, !type !0
@vt2b = constant ptr @vf2b, !type !0

define internal i1 @vf2a(ptr %this) {
  ret i1 0
}

define internal i1 @vf2b(ptr %this) {
  ret i1 1
}

!0 = !{i32 0, !"typeid2"}
