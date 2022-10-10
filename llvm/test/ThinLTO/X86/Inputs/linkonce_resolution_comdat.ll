target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$c2 = comdat any

define linkonce_odr i32 @f(i8*) unnamed_addr comdat($c2) {
    ret i32 41
}

define i32 @g() {
    %i = call i32 @f(i8* null)
    ret i32 %i
}
