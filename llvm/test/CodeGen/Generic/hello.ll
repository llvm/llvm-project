; RUN: llc < %s

@.str_1 = internal constant [7 x i8] c"hello\0A\00"             ; <[7 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
        %s = getelementptr [7 x i8]* @.str_1, i64 0, i64 0              ; <i8*> [#uses=1]
        call i32 (i8*, ...)* @printf( i8* %s )          ; <i32>:1 [#uses=0]
        ret i32 0
}
