; Make sure we don't get an assertion failure, even though this is a parse 
; error
; RUN: not llvm-as %s -o /dev/null -f |& grep {'@foo' defined with}

%ty = type void (i32)

declare %ty* @foo()

define void @test() {
        call %ty* @foo( )               ; <%ty*>:0 [#uses=0]
        ret void
}

