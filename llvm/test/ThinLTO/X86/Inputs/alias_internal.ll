target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @f(ptr) unnamed_addr {
    ret i32 42
}

@a2 = weak alias i32 (ptr), ptr @f
