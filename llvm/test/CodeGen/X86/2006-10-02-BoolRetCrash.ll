; RUN: llvm-as < %s | llc 
; PR933

define fastcc i1 @test() {
        ret i1 true
}

