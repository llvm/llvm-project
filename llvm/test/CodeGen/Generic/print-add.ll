; RUN: llc < %s

@.str_1 = internal constant [4 x i8] c"%d\0A\00"                ; <[4 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
        %f = getelementptr [4 x i8]* @.str_1, i64 0, i64 0              ; <i8*> [#uses=3]
        %d = add i32 1, 0               ; <i32> [#uses=3]
        call i32 (i8*, ...)* @printf( i8* %f, i32 %d )          ; <i32>:1 [#uses=0]
        %e = add i32 38, 2              ; <i32> [#uses=2]
        call i32 (i8*, ...)* @printf( i8* %f, i32 %e )          ; <i32>:2 [#uses=0]
        %g = add i32 %d, %d             ; <i32> [#uses=1]
        %h = add i32 %e, %g             ; <i32> [#uses=1]
        call i32 (i8*, ...)* @printf( i8* %f, i32 %h )          ; <i32>:3 [#uses=0]
        ret i32 0
}

