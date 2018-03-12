target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@baz = internal constant i32 10, align 4

define linkonce_odr i32 @foo() {
  %1 = load i32, i32* @baz, align 4
  ret i32 %1
}
