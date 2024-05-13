target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @foo(ptr %p) {
  %t = load i64, ptr %p
  ret i64 %t
}
