; RUN: llvm-as < %s | llc -march=c | not grep extern.*msg
; PR472

@msg = internal global [6 x i8] c"hello\00"             ; <[6 x i8]*> [#uses=1]

define i8* @foo() {
entry:
        ret i8* getelementptr ([6 x i8]* @msg, i32 0, i32 0)
}

