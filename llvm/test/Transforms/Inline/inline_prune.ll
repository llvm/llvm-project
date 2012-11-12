; RUN: opt < %s -inline -S | \
; RUN:    not grep "callee[12]("
; RUN: opt < %s -inline -S | not grep mul

define internal i32 @callee1(i32 %A, i32 %B) {
        %cond = icmp eq i32 %A, 123             ; <i1> [#uses=1]
        br i1 %cond, label %T, label %F

T:              ; preds = %0
        %C = mul i32 %B, %B             ; <i32> [#uses=1]
        ret i32 %C

F:              ; preds = %0
        ret i32 0
}

define internal i32 @callee2(i32 %A, i32 %B) {
        switch i32 %A, label %T [
                 i32 10, label %F
                 i32 1234, label %G
        ]
                ; No predecessors!
        %cond = icmp eq i32 %A, 123             ; <i1> [#uses=1]
        br i1 %cond, label %T, label %F

T:              ; preds = %1, %0
        %C = mul i32 %B, %B             ; <i32> [#uses=1]
        ret i32 %C

F:              ; preds = %1, %0
        ret i32 0

G:              ; preds = %0
        %D = mul i32 %B, %B             ; <i32> [#uses=1]
        %E = mul i32 %D, %B             ; <i32> [#uses=1]
        ret i32 %E
}

define i32 @test(i32 %A) {
        %X = call i32 @callee1( i32 10, i32 %A )                ; <i32> [#uses=1]
        %Y = call i32 @callee2( i32 10, i32 %A )                ; <i32> [#uses=1]
        %Z = add i32 %X, %Y             ; <i32> [#uses=1]
        ret i32 %Z
}

