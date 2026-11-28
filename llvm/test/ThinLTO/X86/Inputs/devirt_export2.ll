target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A1fEv] }, !type !1

define internal i32 @_ZN1A1fEv(ptr %this) { ret i32 0 }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i64 16, !"_ZTS1A"}
