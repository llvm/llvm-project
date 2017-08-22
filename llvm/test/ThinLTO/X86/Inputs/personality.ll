target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @bar() personality i32 (i32, i32, i64, i8*, i8*)* @personality_routine {
 ret void
}

define protected i32 @personality_routine(i32, i32, i64, i8*, i8*) {
  ret i32 0
}
