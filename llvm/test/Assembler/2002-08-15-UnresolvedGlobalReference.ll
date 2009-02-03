; RUN: llvm-as %s -o /dev/null -f

@.LC0 = internal global [12 x i8] c"hello world\00"             ; <[12 x i8]*> [#uses=1]

define i8* @test() {
        ret i8* getelementptr ([12 x i8]* @.LC0, i64 0, i64 0)
}

