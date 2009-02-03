; Test that any rethrown exceptions in an inlined function are automatically
; turned into branches to the invoke destination.

; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep unwind$

declare void @might_throw()

define internal i32 @callee() {
        invoke void @might_throw( )
                        to label %cont unwind label %exc

cont:           ; preds = %0
        ret i32 0

exc:            ; preds = %0a
       ; This just rethrows the exception!
        unwind
}

; caller returns true if might_throw throws an exception... which gets
; propagated by callee.
define i32 @caller() {
        %X = invoke i32 @callee( )
                        to label %cont unwind label %Handler            ; <i32> [#uses=1]

cont:           ; preds = %0
        ret i32 %X

Handler:                ; preds = %0
; This consumes an exception thrown by might_throw
        ret i32 1
}
