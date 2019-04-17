; RUN: opt < %s -extract-blocks -disable-output
define i32 @foo() personality i32 (...)* @__gcc_personality_v0 {
        br label %EB

EB:             ; preds = %0
        %V = invoke i32 @foo( )
                        to label %Cont unwind label %Unw                ; <i32> [#uses=1]

Cont:           ; preds = %EB
        ret i32 %V

Unw:            ; preds = %EB
        %exn = landingpad { i8*, i32 }
                 catch i8* null
        resume { i8*, i32 } %exn
}

declare i32 @__gcc_personality_v0(...)
