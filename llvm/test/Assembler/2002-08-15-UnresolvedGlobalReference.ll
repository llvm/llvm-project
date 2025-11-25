; RUN: llvm-as  %s -o /dev/null
; RUN: verify-uselistorder  %s

@.LC0 = internal global [12 x i8] c"hello world\00"             ; <ptr> [#uses=1]

define ptr @test() {
        ret ptr @.LC0
}

