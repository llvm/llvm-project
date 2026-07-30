target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$f = comdat any
$g = comdat any

@g_private = private global i32 41, comdat($g)

define linkonce_odr i32 @f(ptr) unnamed_addr comdat($f) {
    ret i32 41
}

define linkonce_odr i32 @g() unnamed_addr comdat($g) {
    ret i32 41
}

define internal void @g_internal() unnamed_addr comdat($g) {
    ret void
}

define i32 @h() {
    %i = call i32 @f(ptr null)
    ret i32 %i
}
