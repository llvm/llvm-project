; Test Case for PR1080
; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@str = internal constant [4 x i8] c"-ga\00"             ; <ptr> [#uses=2]

define i32 @main(i32 %argc, ptr %argv) {
entry:
        %tmp65 = getelementptr ptr, ptr %argv, i32 1                ; <ptr> [#uses=1]
        %tmp66 = load ptr, ptr %tmp65               ; <ptr> [#uses=0]
        %cmp = icmp ne i32 sub (i32 ptrtoint (ptr getelementptr ([4 x i8], ptr @str, i32 0, i64 1) to i32), i32 ptrtoint (ptr @str to i32)), 1
        br i1 %cmp, label %exit_1, label %exit_2

exit_1:         ; preds = %entry
        ret i32 0

exit_2:         ; preds = %entry
        ret i32 1
}

