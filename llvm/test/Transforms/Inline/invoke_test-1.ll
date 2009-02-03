; Test that we can inline a simple function, turning the calls in it into invoke
; instructions

; RUN: llvm-as < %s | opt -inline | llvm-dis | \
; RUN:   not grep {call\[^e\]}

declare void @might_throw()

define internal void @callee() {
        call void @might_throw( )
        ret void
}

; caller returns true if might_throw throws an exception...
define i32 @caller() {
        invoke void @callee( )
                        to label %cont unwind label %exc

cont:           ; preds = %0
        ret i32 0

exc:            ; preds = %0
        ret i32 1
}
