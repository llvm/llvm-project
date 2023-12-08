target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main({ i64, ptr } %unnamed) #0 {
  %1 = call i32 @_simplefunction() #1
  ret i32 %1
}
declare i32 @_simplefunction() #1
