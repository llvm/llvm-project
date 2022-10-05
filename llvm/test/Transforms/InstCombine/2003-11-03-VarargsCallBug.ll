; The cast in this testcase is not eliminable on a 32-bit target!
; RUN: opt < %s -passes=instcombine -S | grep inttoptr

target datalayout = "e-p:32:32"

declare void @foo(...)

define void @test(i64 %X) {
        %Y = inttoptr i64 %X to ptr            ; <ptr> [#uses=1]
        call void (...) @foo( ptr %Y )
        ret void
}

