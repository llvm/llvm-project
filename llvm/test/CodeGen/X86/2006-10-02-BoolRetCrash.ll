; RUN: llc -mtriple=i686 < %s
; PR933

define fastcc i1 @test() {
        ret i1 true
}

