; ModuleID = 'local_name_conflict.o'
source_filename = "local_name_conflict.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @b() {
entry:
  %call = call i32 @foo()
  ret i32 %call
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @foo() {
entry:
  ret i32 2
}
