; RUN: llc < %s
; PR2625

define i32 @main({ i32, { i32 } }*) {
entry:
        %state = alloca { i32, { i32 } }*               ; <{ i32, { i32 } }**> [#uses=2]
        store { i32, { i32 } }* %0, { i32, { i32 } }** %state
        %retval = alloca i32            ; <i32*> [#uses=2]
        store i32 0, i32* %retval
        load { i32, { i32 } }** %state          ; <{ i32, { i32 } }*>:1 [#uses=1]
        store { i32, { i32 } } zeroinitializer, { i32, { i32 } }* %1
        br label %return

return:         ; preds = %entry
        load i32* %retval               ; <i32>:2 [#uses=1]
        ret i32 %2
}
