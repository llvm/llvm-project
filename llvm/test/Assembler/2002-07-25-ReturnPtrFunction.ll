; Test that returning a pointer to a function causes the disassembler to print 
; the right thing.
;
; RUN: llvm-as < %s | llvm-dis | llvm-as

%ty = type void (i32)

declare %ty* @foo()

define void @test() {
        call %ty* ()* @foo( )           ; <%ty*>:1 [#uses=0]
        ret void
}


