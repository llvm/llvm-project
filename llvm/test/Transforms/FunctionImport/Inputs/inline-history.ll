target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @imported_func() {
  call void @another_func(), !inline_history !0
  ret void
}

declare void @another_func()

!0 = !{ptr @another_func}
