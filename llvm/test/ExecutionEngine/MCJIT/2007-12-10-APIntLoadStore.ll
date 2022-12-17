; RUN: %lli -jit-kind=mcjit -force-interpreter %s
; PR1836

define i32 @main() {
entry:
    %retval = alloca i32        ; <ptr> [#uses=2]
    %tmp = alloca i32       ; <ptr> [#uses=2]
    %x = alloca i75, align 16       ; <ptr> [#uses=1]
    %"alloca point" = bitcast i32 0 to i32      ; <i32> [#uses=0]
    store i75 999, ptr %x, align 16
    store i32 0, ptr %tmp, align 4
    %tmp1 = load i32, ptr %tmp, align 4     ; <i32> [#uses=1]
    store i32 %tmp1, ptr %retval, align 4
    br label %return

return:     ; preds = %entry
    %retval2 = load i32, ptr %retval        ; <i32> [#uses=1]
    ret i32 %retval2
}
