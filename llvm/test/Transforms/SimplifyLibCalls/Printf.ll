; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep putchar
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*printf}

@str = internal constant [13 x i8] c"hello world\0A\00"         ; <[13 x i8]*> [#uses=1]
@str1 = internal constant [2 x i8] c"h\00"              ; <[2 x i8]*> [#uses=1]

define void @foo() {
entry:
        %tmp1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([13 x i8]* @str, i32 0, i32 0) )         ; <i32> [#uses=0]
        ret void
}

declare i32 @printf(i8*, ...)

define void @bar() {
entry:
        %tmp1 = tail call i32 (i8*, ...)* @printf( i8* getelementptr ([2 x i8]* @str1, i32 0, i32 0) )         ; <i32> [#uses=0]
        ret void
}

