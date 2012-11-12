; Test that we can inline a simple function, turning the calls in it into invoke
; instructions

; RUN: opt < %s -inline -S | \
; RUN:   not grep "call[^e]"

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
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
        ret i32 1
}

declare i32 @__gxx_personality_v0(...)
