; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | \
; RUN:   not grep {call void @DEAD}

; Test that we can thread a simple known condition through switch statements.

declare void @foo1()

declare void @foo2()

declare void @DEAD()

define void @test1(i32 %V) {
        switch i32 %V, label %A [
                 i32 4, label %T
                 i32 17, label %Done
                 i32 1234, label %A
        ]
;; V == 4 if we get here.
T:              ; preds = %0
        call void @foo1( )
        ;; This switch is always statically determined.
        switch i32 %V, label %A2 [
                 i32 4, label %B
                 i32 17, label %C
                 i32 42, label %C
        ]
A2:             ; preds = %T
        call void @DEAD( )
        call void @DEAD( )
        ;; always true
        %cond2 = icmp eq i32 %V, 4              ; <i1> [#uses=1]
        br i1 %cond2, label %Done, label %C
A:              ; preds = %0, %0
        call void @foo1( )
        ;; always true
        %cond = icmp ne i32 %V, 4               ; <i1> [#uses=1]
        br i1 %cond, label %Done, label %C
Done:           ; preds = %B, %A, %A2, %0
        ret void
B:              ; preds = %T
        call void @foo2( )
        ;; always true
        %cond3 = icmp eq i32 %V, 4              ; <i1> [#uses=1]
        br i1 %cond3, label %Done, label %C
C:              ; preds = %B, %A, %A2, %T, %T
        call void @DEAD( )
        ret void
}

define void @test2(i32 %V) {
        switch i32 %V, label %A [
                 i32 4, label %T
                 i32 17, label %D
                 i32 1234, label %E
        ]
;; V != 4, 17, 1234 here.
A:              ; preds = %0
        call void @foo1( )
        ;; This switch is always statically determined.
        switch i32 %V, label %E [
                 i32 4, label %C
                 i32 17, label %C
                 i32 42, label %D
        ]
;; unreacahble.
C:              ; preds = %A, %A
        call void @DEAD( )
        ret void
T:              ; preds = %0
        call void @foo1( )
        call void @foo1( )
        ret void
D:              ; preds = %A, %0
        call void @foo1( )
        ret void
E:              ; preds = %A, %0
        ret void
}

