; RUN: opt < %s -basic-aa -gvn -instcombine -S | FileCheck %s

declare i32* @test(i32* nocapture)

define i32 @test2() {
; CHECK: ret i32 0
       %P = alloca i32
       %Q = call i32* @test(i32* %P)
       %a = load i32, i32* %P
       store i32 4, i32* %Q   ;; cannot clobber P since it is nocapture.
       %b = load i32, i32* %P
       %c = sub i32 %a, %b
       ret i32 %c
}

declare void @test3(i32** %p, i32* %q) nounwind

define i32 @test4(i32* noalias nocapture %p) nounwind {
; CHECK: call void @test3
; CHECK: store i32 0, i32* %p
; CHECK: store i32 1, i32* %x
; CHECK: %y = load i32, i32* %p
; CHECK: ret i32 %y
entry:
       %q = alloca i32*
       ; Here test3 might store %p to %q. This doesn't violate %p's nocapture
       ; attribute since the copy doesn't outlive the function.
       call void @test3(i32** %q, i32* %p) nounwind
       store i32 0, i32* %p
       %x = load i32*, i32** %q
       ; This store might write to %p and so we can't eliminate the subsequent
       ; load
       store i32 1, i32* %x
       %y = load i32, i32* %p
       ret i32 %y
}
