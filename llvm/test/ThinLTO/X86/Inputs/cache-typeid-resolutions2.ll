target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vt2 = constant i1 (i8*)* @vf2, !type !0

define internal i1 @vf2(i8* %this) {
  ret i1 0
}

!0 = !{i32 0, !"typeid2"}
