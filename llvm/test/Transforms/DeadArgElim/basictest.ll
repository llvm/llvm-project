; RUN: opt < %s -passes=deadargelim -S | not grep DEADARG

; test - an obviously dead argument
define internal i32 @test(i32 %v, i32 %DEADARG1, i32* %p) {
        store i32 %v, i32* %p
        ret i32 %v
}

; hardertest - an argument which is only used by a call of a function with a 
; dead argument.
define internal i32 @hardertest(i32 %DEADARG2) {
        %p = alloca i32         ; <i32*> [#uses=1]
        %V = call i32 @test( i32 5, i32 %DEADARG2, i32* %p )            ; <i32> [#uses=1]
        ret i32 %V
}

; evenhardertest - recursive dead argument...
define internal void @evenhardertest(i32 %DEADARG3) {
        call void @evenhardertest( i32 %DEADARG3 )
        ret void
}

define internal void @needarg(i32 %TEST) {
        call i32 @needarg2( i32 %TEST )         ; <i32>:1 [#uses=0]
        ret void
}

define internal i32 @needarg2(i32 %TEST) {
        ret i32 %TEST
}

define internal void @needarg3(i32 %TEST3) {
        call void @needarg( i32 %TEST3 )
        ret void
}

