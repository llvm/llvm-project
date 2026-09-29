; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s

declare ptr @test(ptr nocapture)

define i32 @test2() {
; CHECK: ret i32 0
       %P = alloca i32
       %Q = call ptr @test(ptr %P)
       %a = load i32, ptr %P
       store i32 4, ptr %Q   ;; cannot clobber P since it is nocapture.
       %b = load i32, ptr %P
       %c = sub i32 %a, %b
       ret i32 %c
}

declare void @test3(ptr %p, ptr %q) nounwind

define i32 @test4(ptr noalias nocapture %p) nounwind {
; CHECK: call void @test3
; CHECK: store i32 0, ptr %p
; CHECK: store i32 1, ptr %x
; CHECK: %y = load i32, ptr %p
; CHECK: ret i32 %y
entry:
       %q = alloca ptr
       ; Here test3 might store %p to %q. This doesn't violate %p's nocapture
       ; attribute since the copy doesn't outlive the function.
       call void @test3(ptr %q, ptr %p) nounwind
       store i32 0, ptr %p
       %x = load ptr, ptr %q
       ; This store might write to %p and so we can't eliminate the subsequent
       ; load
       store i32 1, ptr %x
       %y = load i32, ptr %p
       ret i32 %y
}
