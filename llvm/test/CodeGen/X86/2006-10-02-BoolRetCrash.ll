; RUN: llc < %s 
; PR933
; REQUIRES: default_triple

define fastcc i1 @test() {
        ret i1 true
}

