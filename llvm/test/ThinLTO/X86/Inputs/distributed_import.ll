target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@G = internal global i32 7
define i32 @g() {
entry:
  %0 = load i32, ptr @G
  ret i32 %0
}

@analias = alias void (...), ptr @aliasee
define void @aliasee() {
entry:
      ret void
}
