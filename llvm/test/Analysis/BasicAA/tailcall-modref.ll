; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s

define i32 @test() {
; CHECK: ret i32 0
        %A = alloca i32         ; <ptr> [#uses=3]
        call void @foo( ptr %A )
        %X = load i32, ptr %A               ; <i32> [#uses=1]
        tail call void @bar( )
        %Y = load i32, ptr %A               ; <i32> [#uses=1]
        %Z = sub i32 %X, %Y             ; <i32> [#uses=1]
        ret i32 %Z
}

declare void @foo(ptr)

declare void @bar()
