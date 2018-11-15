; ModuleID = 'b.cpp'
source_filename = "b.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline optnone uwtable
define void @_Z5Alphav() {
entry:
  call void @_Z5Bravov()
  ret void
}

declare void @_Z5Bravov()
