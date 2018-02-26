target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@G = internal global i32 7
define i32 @g() {
entry:
  %0 = load i32, i32* @G
  ret i32 %0
}

@analias = alias void (...), bitcast (void ()* @aliasee to void (...)*)
define void @aliasee() {
entry:
      ret void
}
