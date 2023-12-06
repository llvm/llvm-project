target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @m1_f1()

define dso_local void @m3_f1() {
  call void @m1_f1()
  ret void
}
