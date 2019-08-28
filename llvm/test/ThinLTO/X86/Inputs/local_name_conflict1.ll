; ModuleID = 'local_name_conflict.o'
source_filename = "local_name_conflict.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@baz = internal constant i32 10, align 4

; Function Attrs: noinline nounwind uwtable
define i32 @a() {
entry:
  %call = call i32 @foo()
  ret i32 %call
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @foo() {
entry:
  %0 = load i32, i32* @baz, align 4
  ret i32 %0
}
