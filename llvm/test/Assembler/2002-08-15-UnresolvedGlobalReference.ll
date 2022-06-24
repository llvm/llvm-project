; RUN: llvm-as --opaque-pointers=0 %s -o /dev/null
; RUN: verify-uselistorder --opaque-pointers=0 %s

@.LC0 = internal global [12 x i8] c"hello world\00"             ; <[12 x i8]*> [#uses=1]

define i8* @test() {
        ret i8* getelementptr ([12 x i8], [12 x i8]* @.LC0, i64 0, i64 0)
}

