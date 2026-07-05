target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
declare i32 @g()
define i32 @main() {
  call i32 @g()
  ret i32 0
}
