; RUN: llc < %s
; PR2625

define i32 @main(ptr) {
entry:
        %state = alloca ptr               ; <ptr> [#uses=2]
        store ptr %0, ptr %state
        %retval = alloca i32            ; <ptr> [#uses=2]
        store i32 0, ptr %retval
        load ptr, ptr %state          ; <ptr>:1 [#uses=1]
        store { i32, { i32 } } zeroinitializer, ptr %1
        br label %return

return:         ; preds = %entry
        load i32, ptr %retval               ; <i32>:2 [#uses=1]
        ret i32 %2
}
