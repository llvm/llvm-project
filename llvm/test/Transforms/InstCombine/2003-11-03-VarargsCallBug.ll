; The cast in this testcase is not eliminable on a 32-bit target!
; RUN: opt < %s -instcombine -S | grep inttoptr

target datalayout = "e-p:32:32"

declare void @foo(...)

define void @test(i64 %X) {
        %Y = inttoptr i64 %X to i32*            ; <i32*> [#uses=1]
        call void (...) @foo( i32* %Y )
        ret void
}

