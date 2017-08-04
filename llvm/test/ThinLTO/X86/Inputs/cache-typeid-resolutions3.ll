target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vt2a = constant i1 (i8*)* @vf2a, !type !0
@vt2b = constant i1 (i8*)* @vf2b, !type !0

define internal i1 @vf2a(i8* %this) {
  ret i1 0
}

define internal i1 @vf2b(i8* %this) {
  ret i1 1
}

!0 = !{i32 0, !"typeid2"}
