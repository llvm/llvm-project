target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare void @m1_variant()

define dso_local void @m1_f1() {
  call void @m1_f2()
  call void @noninterposable_f()
  ret void
}

define internal void @m1_f2() {
  call void @interposable_f()
  ret void
}

define linkonce void @interposable_f() {
  call void @m1_variant()
  ret void
}

define linkonce_odr void @noninterposable_f() {
  call void @m1_variant()
  ret void
}