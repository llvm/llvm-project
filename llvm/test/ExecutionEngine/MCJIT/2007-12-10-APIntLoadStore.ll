; RUN: %lli -force-interpreter %s
; PR1836

define i32 @main() {
entry:
    %retval = alloca i32        ; <i32*> [#uses=2]
    %tmp = alloca i32       ; <i32*> [#uses=2]
    %x = alloca i75, align 16       ; <i75*> [#uses=1]
    %"alloca point" = bitcast i32 0 to i32      ; <i32> [#uses=0]
    store i75 999, i75* %x, align 16
    store i32 0, i32* %tmp, align 4
    %tmp1 = load i32, i32* %tmp, align 4     ; <i32> [#uses=1]
    store i32 %tmp1, i32* %retval, align 4
    br label %return

return:     ; preds = %entry
    %retval2 = load i32, i32* %retval        ; <i32> [#uses=1]
    ret i32 %retval2
}
