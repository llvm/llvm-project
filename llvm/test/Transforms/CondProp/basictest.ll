; RUN: llvm-as < %s | opt -condprop | llvm-dis | \
; RUN:    not grep {br label}


define i32 @test(i1 %C) {
        br i1 %C, label %T1, label %F1

T1:             ; preds = %0
        br label %Cont

F1:             ; preds = %0
        br label %Cont

Cont:           ; preds = %F1, %T1
        %C2 = phi i1 [ false, %F1 ], [ true, %T1 ]              ; <i1> [#uses=1]
        br i1 %C2, label %T2, label %F2

T2:             ; preds = %Cont
        call void @bar( )
        ret i32 17

F2:             ; preds = %Cont
        ret i32 1
}

declare void @bar()

