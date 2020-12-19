; RUN: opt < %s -simplifycfg -S | \
; RUN:   not grep "icmp eq"

; Check that simplifycfg deletes a dead 'seteq' instruction when it
; folds a conditional branch into a switch instruction.

declare void @foo()

declare void @bar()

define void @testcfg(i32 %V) {
        %C = icmp eq i32 %V, 18         ; <i1> [#uses=1]
        %D = icmp eq i32 %V, 180                ; <i1> [#uses=1]
        %E = or i1 %C, %D               ; <i1> [#uses=1]
        br i1 %E, label %L1, label %Sw
Sw:             ; preds = %0
        switch i32 %V, label %L1 [
                 i32 15, label %L2
                 i32 16, label %L2
        ]
L1:             ; preds = %Sw, %0
        call void @foo( )
        ret void
L2:             ; preds = %Sw, %Sw
        call void @bar( )
        ret void
}

