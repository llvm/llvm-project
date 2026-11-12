source_filename = "common2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@P = external global ptr, align 8

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  %0 = load ptr, ptr @P, align 8
  %call = call ptr (...) %0()
  ret void
}
