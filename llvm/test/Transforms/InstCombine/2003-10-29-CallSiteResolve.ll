; RUN: opt < %s -passes=instcombine -disable-output

declare ptr @bar()

define ptr @foo() personality ptr @__gxx_personality_v0 {
        %tmp.11 = invoke ptr @bar( )
                        to label %invoke_cont unwind label %X           ; <ptr> [#uses=1]

invoke_cont:            ; preds = %0
        ret ptr %tmp.11

X:              ; preds = %0
        %exn = landingpad {ptr, i32}
                 cleanup
        ret ptr null
}

declare i32 @__gxx_personality_v0(...)
