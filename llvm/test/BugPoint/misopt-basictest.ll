; RUN: bugpoint %s -dce -bugpoint-deletecalls -simplifycfg -silence-passes

@.LC0 = internal global [13 x i8] c"Hello World\0A\00"          ; <[13 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
        call i32 (i8*, ...)* @printf( i8* getelementptr ([13 x i8]* @.LC0, i64 0, i64 0) )            ; <i32>:1 [#uses=0]
        ret i32 0
}
