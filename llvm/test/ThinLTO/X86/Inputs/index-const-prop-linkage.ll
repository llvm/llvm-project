target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g1 = common global i32 0, align 4
@g2 = global i32 42, align 4
@g3 = available_externally global i32 42, align 4

define i32 @foo() {
  %v1 = load i32, i32* @g1
  %v2 = load i32, i32* @g2
  %v3 = load i32, i32* @g3
  %s1 = add i32 %v1, %v2
  %s2 = add i32 %s1, %v3
  ret i32 %s2
}
