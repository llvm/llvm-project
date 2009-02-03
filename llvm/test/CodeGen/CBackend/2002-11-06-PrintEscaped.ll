; RUN: llvm-as < %s | llc -march=c

@testString = internal constant [18 x i8] c"Escaped newline\5Cn\00"             ; <[18 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
        call i32 (i8*, ...)* @printf( i8* getelementptr ([18 x i8]* @testString, i64 0, i64 0) )                ; <i32>:1 [#uses=0]
        ret i32 0
}

