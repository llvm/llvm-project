target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main({ i64, ptr } %unnamed) #0 {
  %1 = call i32 @func1() #1
  %2 = call i32 @func2() #1
  %3 = call i32 @func3() #1
  %4 = call i32 @callglobalfunc() #1
  %5 = call i32 @callweakfunc() #1
  ret i32 %1
}
declare i32 @func1() #1
declare i32 @func2() #1
declare i32 @func3() #1
declare i32 @callglobalfunc() #1
declare i32 @callweakfunc() #1
