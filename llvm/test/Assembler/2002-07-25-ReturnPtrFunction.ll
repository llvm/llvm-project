; Test that returning a pointer to a function causes the disassembler to print 
; the right thing.
;
; RUN: llvm-as < %s | llvm-dis | llvm-as
; RUN: verify-uselistorder %s

declare ptr @foo()

define void @test() {
        call ptr () @foo( )           ; <ptr>:1 [#uses=0]
        ret void
}


