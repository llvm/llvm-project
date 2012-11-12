; RUN: opt < %s -adce -simplifycfg -S | grep call
declare void @exit(i32)

define i32 @main(i32 %argc) {
        %C = icmp eq i32 %argc, 1               ; <i1> [#uses=2]
        br i1 %C, label %Cond, label %Done

Cond:           ; preds = %0
        br i1 %C, label %Loop, label %Done

Loop:           ; preds = %Loop, %Cond
        call void @exit( i32 0 )
        br label %Loop

Done:           ; preds = %Cond, %0
        ret i32 1
}

