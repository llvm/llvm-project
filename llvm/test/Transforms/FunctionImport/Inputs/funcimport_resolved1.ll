target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @foo() {
  call void @linkonceodrfunc()
  call void @linkonceodrfunc2()
  ret void
}

define linkonce_odr void @linkonceodrfunc() {
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  ret void
}

define linkonce_odr void @linkonceodrfunc2() {
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  call void @f()
  ret void
}

define internal void @f() {
  ret void
}
