target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @f(i8*) unnamed_addr {
    ret i32 42
}

@a2 = weak alias i32 (i8*), i32 (i8*)* @f
