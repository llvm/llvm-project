; RUN: opt < %s -simplifycfg -S | \
; RUN:   not grep switch

; Test normal folding
define i32 @test1() {
        switch i32 5, label %Default [
                 i32 0, label %Foo
                 i32 1, label %Bar
                 i32 2, label %Baz
                 i32 5, label %TheDest
        ]
Default:                ; preds = %0
        ret i32 -1
Foo:            ; preds = %0
        ret i32 -2
Bar:            ; preds = %0
        ret i32 -3
Baz:            ; preds = %0
        ret i32 -4
TheDest:                ; preds = %0
        ret i32 1234
}

; Test folding to default dest
define i32 @test2() {
        switch i32 3, label %Default [
                 i32 0, label %Foo
                 i32 1, label %Bar
                 i32 2, label %Baz
                 i32 5, label %TheDest
        ]
Default:                ; preds = %0
        ret i32 1234
Foo:            ; preds = %0
        ret i32 -2
Bar:            ; preds = %0
        ret i32 -5
Baz:            ; preds = %0
        ret i32 -6
TheDest:                ; preds = %0
        ret i32 -8
}

; Test folding all to same dest
define i32 @test3(i1 %C) {
        br i1 %C, label %Start, label %TheDest
Start:          ; preds = %0
        switch i32 3, label %TheDest [
                 i32 0, label %TheDest
                 i32 1, label %TheDest
                 i32 2, label %TheDest
                 i32 5, label %TheDest
        ]
TheDest:                ; preds = %Start, %Start, %Start, %Start, %Start, %0
        ret i32 1234
}

; Test folding switch -> branch
define i32 @test4(i32 %C) {
        switch i32 %C, label %L1 [
                 i32 0, label %L2
        ]
L1:             ; preds = %0
        ret i32 0
L2:             ; preds = %0
        ret i32 1
}

; Can fold into a cond branch!
define i32 @test5(i32 %C) {
        switch i32 %C, label %L1 [
                 i32 0, label %L2
                 i32 123, label %L1
        ]
L1:             ; preds = %0, %0
        ret i32 0
L2:             ; preds = %0
        ret i32 1
}

