; RUN: llvm-as < %s | llc | grep mflr | count 1

target datalayout = "e-p:32:32"
target triple = "powerpc-apple-darwin8"
@str = internal constant [18 x i8] c"hello world!, %d\0A\00"            ; <[18 x i8]*> [#uses=1]


define i32 @main() {
entry:
        %tmp = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([18 x i8]* @str, i32 0, i32 0) )                ; <i32> [#uses=0]
        ret i32 0
}

declare i32 @printf(i8*, ...)
