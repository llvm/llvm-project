; RUN: opt < %s -adce -disable-output

define void @test() personality i32 (...)* @__gxx_personality_v0 {
        br i1 false, label %then, label %endif

then:           ; preds = %0
        invoke void null( i8* null )
                        to label %invoke_cont unwind label %invoke_catch

invoke_catch:           ; preds = %then
        %exn = landingpad {i8*, i32}
                 cleanup
        resume { i8*, i32 } %exn

invoke_cont:            ; preds = %then
        ret void

endif:          ; preds = %0
        ret void
}

declare i32 @__gxx_personality_v0(...)
