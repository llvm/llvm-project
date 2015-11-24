target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

declare void @g()

define void @f() {
 entry:
  call void @g()
  call void @g()
  call void @g()
  ret void
}
