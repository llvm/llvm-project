; This checks to ensure that the inline pass deletes functions if they get 
; inlined into all of their callers.

; RUN: llvm-as < %s | opt -inline | llvm-dis | \
; RUN:   not grep @reallysmall

define internal i32 @reallysmall(i32 %A) {
        ret i32 %A
}

define void @caller1() {
        call i32 @reallysmall( i32 5 )          ; <i32>:1 [#uses=0]
        ret void
}

define void @caller2(i32 %A) {
        call i32 @reallysmall( i32 %A )         ; <i32>:1 [#uses=0]
        ret void
}

define i32 @caller3(i32 %A) {
        %B = call i32 @reallysmall( i32 %A )            ; <i32> [#uses=1]
        ret i32 %B
}

