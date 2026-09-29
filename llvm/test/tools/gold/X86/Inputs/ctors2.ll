target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @bar, ptr null }]

define void @bar() {
  ret void
}
