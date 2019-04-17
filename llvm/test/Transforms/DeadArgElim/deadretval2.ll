; RUN: opt < %s -deadargelim -die -S > %t
; RUN: cat %t | not grep DEAD
; RUN: cat %t | grep LIVE | count 4

@P = external global i32                ; <i32*> [#uses=1]

; Dead arg only used by dead retval
define internal i32 @test(i32 %DEADARG) {
        ret i32 %DEADARG
}

define internal i32 @test2(i32 %DEADARG) {
        %DEADRETVAL = call i32 @test( i32 %DEADARG )            ; <i32> [#uses=1]
        ret i32 %DEADRETVAL
}

define void @test3(i32 %X) {
        %DEADRETVAL = call i32 @test2( i32 %X )         ; <i32> [#uses=0]
        ret void
}

define internal i32 @foo() {
        %DEAD = load i32, i32* @P            ; <i32> [#uses=1]
        ret i32 %DEAD
}

define internal i32 @id(i32 %X) {
        ret i32 %X
}

define void @test4() {
        %DEAD = call i32 @foo( )                ; <i32> [#uses=1]
        %DEAD2 = call i32 @id( i32 %DEAD )              ; <i32> [#uses=0]
        ret void
}

; These test if returning another functions return value properly marks that
; other function's return value as live. We do this twice, with the functions in
; different orders (ie, first the caller, than the callee and first the callee
; and then the caller) since DAE processes functions one by one and handles
; these cases slightly different.

define internal i32 @test5() {
  ret i32 123 
}

define i32 @test6() {
  %LIVE = call i32 @test5()
  ret i32 %LIVE
}

define i32 @test7() {
  %LIVE = call i32 @test8()
  ret i32 %LIVE
}

define internal i32 @test8() {
  ret i32 124
}
