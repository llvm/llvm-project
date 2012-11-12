; RUN: llc < %s

@.str_1 = internal constant [4 x i8] c"%d\0A\00"                ; <[4 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
        %f = getelementptr [4 x i8]* @.str_1, i64 0, i64 0              ; <i8*> [#uses=1]
        %d = add i32 0, 0               ; <i32> [#uses=1]
        %tmp.0 = call i32 (i8*, ...)* @printf( i8* %f, i32 %d )         ; <i32> [#uses=0]
        ret i32 0
}

