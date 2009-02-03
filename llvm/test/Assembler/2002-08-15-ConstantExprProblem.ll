; RUN: llvm-as %s -o /dev/null -f

@.LC0 = internal global [12 x i8] c"hello world\00"             ; <[12 x i8]*> [#uses=1]

define i8* @test() {
; <label>:0
        br label %BB1

BB1:            ; preds = %BB2, %0
        %ret = phi i8* [ getelementptr ([12 x i8]* @.LC0, i64 0, i64 0), %0 ], [ null, %BB2 ]          ; <i8*> [#uses=1]
        ret i8* %ret

BB2:            ; No predecessors!
        br label %BB1
}

