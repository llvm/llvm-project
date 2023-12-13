target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f() {
entry:
  call void @f_1();
  ret void
}

declare void @f_1();
