; RUN: opt < %s -passes=extract-blocks -disable-output
define i32 @foo() personality ptr @__gcc_personality_v0 {
        br label %EB

EB:             ; preds = %0
        %V = invoke i32 @foo( )
                        to label %Cont unwind label %Unw                ; <i32> [#uses=1]

Cont:           ; preds = %EB
        ret i32 %V

Unw:            ; preds = %EB
        %exn = landingpad { ptr, i32 }
                 catch ptr null
        resume { ptr, i32 } %exn
}

declare i32 @__gcc_personality_v0(...)
