; This test makes sure that these instructions are properly eliminated.
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep phi

define i32 @test1(i32 %A, i1 %b) {
BB0:
        br i1 %b, label %BB1, label %BB2

BB1:            ; preds = %BB0
        ; Combine away one argument PHI nodes
        %B = phi i32 [ %A, %BB0 ]               ; <i32> [#uses=1]
        ret i32 %B

BB2:            ; preds = %BB0
        ret i32 %A
}

define i32 @test2(i32 %A, i1 %b) {
BB0:
        br i1 %b, label %BB1, label %BB2

BB1:            ; preds = %BB0
        br label %BB2

BB2:            ; preds = %BB1, %BB0
        ; Combine away PHI nodes with same values
        %B = phi i32 [ %A, %BB0 ], [ %A, %BB1 ]         ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test3(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ %B, %Loop ]                ; <i32> [#uses=2]
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 %B
}

define i32 @test4(i1 %b) {
BB0:
        ; Loop is unreachable
        ret i32 7

Loop:           ; preds = %L2, %Loop
        ; PHI has same value always.
        %B = phi i32 [ %B, %L2 ], [ %B, %Loop ]         ; <i32> [#uses=2]
        br i1 %b, label %L2, label %Loop

L2:             ; preds = %Loop
        br label %Loop
}

define i32 @test5(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI has same value always.
        %B = phi i32 [ %A, %BB0 ], [ undef, %Loop ]             ; <i32> [#uses=1]
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 %B
}

define i32 @test6(i32 %A, i1 %b) {
BB0:
        %X = bitcast i32 %A to i32              ; <i32> [#uses=1]
        br i1 %b, label %BB1, label %BB2

BB1:            ; preds = %BB0
        %Y = bitcast i32 %A to i32              ; <i32> [#uses=1]
        br label %BB2

BB2:            ; preds = %BB1, %BB0
        ;; Suck casts into phi
        %B = phi i32 [ %X, %BB0 ], [ %Y, %BB1 ]         ; <i32> [#uses=1]
        ret i32 %B
}

define i32 @test7(i32 %A, i1 %b) {
BB0:
        br label %Loop

Loop:           ; preds = %Loop, %BB0
        ; PHI is dead.
        %B = phi i32 [ %A, %BB0 ], [ %C, %Loop ]                ; <i32> [#uses=1]
        %C = add i32 %B, 123            ; <i32> [#uses=1]
        br i1 %b, label %Loop, label %Exit

Exit:           ; preds = %Loop
        ret i32 0
}

define i32* @test8({ i32, i32 } *%A, i1 %b) {
BB0:
        %X = getelementptr { i32, i32 } *%A, i32 0, i32 1
        br i1 %b, label %BB1, label %BB2

BB1:
        %Y = getelementptr { i32, i32 } *%A, i32 0, i32 1
        br label %BB2

BB2:
        ;; Suck GEPs into phi
        %B = phi i32* [ %X, %BB0 ], [ %Y, %BB1 ]
        ret i32* %B
}


