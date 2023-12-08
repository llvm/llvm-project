target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f1() {
  call void @linkonce_odr()
  ret void
}

define linkonce_odr void @linkonce_odr() {
  ret void
}
