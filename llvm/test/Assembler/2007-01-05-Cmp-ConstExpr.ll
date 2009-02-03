; Test Case for PR1080
; RUN: llvm-as %s -o /dev/null -f

@str = internal constant [4 x i8] c"-ga\00"             ; <[4 x i8]*> [#uses=2]

define i32 @main(i32 %argc, i8** %argv) {
entry:
        %tmp65 = getelementptr i8** %argv, i32 1                ; <i8**> [#uses=1]
        %tmp66 = load i8** %tmp65               ; <i8*> [#uses=0]
        br i1 icmp ne (i32 sub (i32 ptrtoint (i8* getelementptr ([4 x i8]* @str, i32 0, i64 1) to i32), i32 ptrtoint ([4 x i8]* @str to i32)), i32 1), label %exit_1, label %exit_2

exit_1:         ; preds = %entry
        ret i32 0

exit_2:         ; preds = %entry
        ret i32 1
}

