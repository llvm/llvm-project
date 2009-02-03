; RUN: llvm-as < %s | llc

@.str_1 = internal constant [42 x i8] c"   ui = %u (0x%x)\09\09UL-ui = %lld (0x%llx)\0A\00"             ; <[42 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define internal i64 @getL() {
entry:
        ret i64 -5787213826675591005
}

define i32 @main(i32 %argc.1, i8** %argv.1) {
entry:
        %tmp.11 = call i64 @getL( )             ; <i64> [#uses=2]
        %tmp.5 = trunc i64 %tmp.11 to i32               ; <i32> [#uses=2]
        %tmp.23 = and i64 %tmp.11, -4294967296          ; <i64> [#uses=2]
        %tmp.16 = call i32 (i8*, ...)* @printf( i8* getelementptr ([42 x i8]* @.str_1, i64 0, i64 0), i32 %tmp.5, i32 %tmp.5, i64 %tmp.23, i64 %tmp.23 )              ; <i32> [#uses=0]
        ret i32 0
}

