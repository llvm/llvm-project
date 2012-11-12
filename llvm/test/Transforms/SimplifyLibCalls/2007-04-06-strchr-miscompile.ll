; PR1307
; RUN: opt < %s -simplify-libcalls -instcombine -S > %t
; RUN: grep "@str,.*i64 3" %t
; RUN: grep "@str1,.*i64 7" %t
; RUN: grep "ret i8.*null" %t
; END.

@str = internal constant [5 x i8] c"foog\00"
@str1 = internal constant [8 x i8] c"blahhh!\00"
@str2 = internal constant [5 x i8] c"Ponk\00"

define i8* @test1() {
        %tmp3 = tail call i8* @strchr( i8* getelementptr ([5 x i8]* @str, i32 0, i32 2), i32 103 )              ; <i8*> [#uses=1]
        ret i8* %tmp3
}

declare i8* @strchr(i8*, i32)

define i8* @test2() {
        %tmp3 = tail call i8* @strchr( i8* getelementptr ([8 x i8]* @str1, i32 0, i32 2), i32 0 )               ; <i8*> [#uses=1]
        ret i8* %tmp3
}

define i8* @test3() {
entry:
        %tmp3 = tail call i8* @strchr( i8* getelementptr ([5 x i8]* @str2, i32 0, i32 1), i32 80 )              ; <i8*> [#uses=1]
        ret i8* %tmp3
}

