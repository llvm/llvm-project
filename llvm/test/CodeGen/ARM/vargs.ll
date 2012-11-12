; RUN: llc < %s -march=arm
@str = internal constant [43 x i8] c"Hello World %d %d %d %d %d %d %d %d %d %d\0A\00"           ; <[43 x i8]*> [#uses=1]

define i32 @main() {
entry:
        %tmp = call i32 (i8*, ...)* @printf( i8* getelementptr ([43 x i8]* @str, i32 0, i64 0), i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10 )         ; <i32> [#uses=0]
        %tmp2 = call i32 (i8*, ...)* @printf( i8* getelementptr ([43 x i8]* @str, i32 0, i64 0), i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1 )                ; <i32> [#uses=0]
        ret i32 11
}

declare i32 @printf(i8*, ...)

