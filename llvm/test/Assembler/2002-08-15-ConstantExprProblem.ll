; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@.LC0 = internal global [12 x i8] c"hello world\00"             ; <ptr> [#uses=1]

define ptr @test() {
; <label>:0
        br label %BB1

BB1:            ; preds = %BB2, %0
        %ret = phi ptr [ @.LC0, %0 ], [ null, %BB2 ]          ; <ptr> [#uses=1]
        ret ptr %ret

BB2:            ; No predecessors!
        br label %BB1
}

