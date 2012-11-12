; RUN: opt < %s -inline -disable-output

declare i32 @External()

define internal i32 @Callee() {
        %I = call i32 @External( )              ; <i32> [#uses=2]
        %J = add i32 %I, %I             ; <i32> [#uses=1]
        ret i32 %J
}

define i32 @Caller() {
        %V = invoke i32 @Callee( )
                        to label %Ok unwind label %Bad          ; <i32> [#uses=1]

Ok:             ; preds = %0
        ret i32 %V

Bad:            ; preds = %0
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
        ret i32 0
}

declare i32 @__gxx_personality_v0(...)

