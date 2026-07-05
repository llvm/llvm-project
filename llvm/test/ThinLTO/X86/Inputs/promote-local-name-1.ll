source_filename = "llvm/test/LTO/X86/promote-local-name-1.ll"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@baz = internal constant i32 10, align 4

; Function Attrs: noinline nounwind uwtable
define i32 @b() {
entry:
  %call = call i32 @foo()
  ret i32 %call
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @foo() {
entry:
  %0 = load i32, ptr @baz, align 4
  ret i32 %0
}
