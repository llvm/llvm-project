; Test that if an invoked function is inlined, and if that function cannot
; throw, that the dead handler is now unreachable.

; RUN: llvm-as < %s | opt -inline -simplifycfg | llvm-dis | \
; RUN:   not grep UnreachableExceptionHandler

declare void @might_throw()

define internal i32 @callee() {
        invoke void @might_throw( )
                        to label %cont unwind label %exc

cont:           ; preds = %0
        ret i32 0

exc:            ; preds = %0
        ret i32 1
}

; caller returns true if might_throw throws an exception... callee cannot throw.
define i32 @caller() {
        %X = invoke i32 @callee( )
                        to label %cont unwind label %UnreachableExceptionHandler                ; <i32> [#uses=1]

cont:           ; preds = %0
        ret i32 %X

UnreachableExceptionHandler:            ; preds = %0
        ret i32 -1
}
