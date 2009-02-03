; RUN: llvm-as < %s | opt -condprop | llvm-dis | not grep phi

define i32 @test(i32 %C, i1 %Val) {
        switch i32 %C, label %T1 [
                 i32 4, label %T2
                 i32 17, label %T3
        ]

T1:             ; preds = %0
        call void @a( )
        br label %Cont

T2:             ; preds = %0
        call void @b( )
        br label %Cont

T3:             ; preds = %0
        call void @c( )
        br label %Cont

Cont:           ; preds = %T3, %T2, %T1
        ;; PHI becomes dead after threading T2
        %C2 = phi i1 [ %Val, %T1 ], [ true, %T2 ], [ %Val, %T3 ]                ; <i1> [#uses=1]
        br i1 %C2, label %L2, label %F2

L2:             ; preds = %Cont
        call void @d( )
        ret i32 17

F2:             ; preds = %Cont
        call void @e( )
        ret i32 1
}

declare void @a()

declare void @b()

declare void @c()

declare void @d()

declare void @e()
