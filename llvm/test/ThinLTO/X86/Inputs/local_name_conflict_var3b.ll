; ModuleID = 'local_name_conflict_var.o'
source_filename = "local_name_conflict_var.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @baz() {
  ret i32 0
}

define i32 @c() {
entry:
  %call1 = call i32 (...) @baz()
  ret i32 0
}
