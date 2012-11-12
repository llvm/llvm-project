; RUN: opt < %s -instcombine -disable-output

declare i32* @bar()

define float* @foo() {
        %tmp.11 = invoke float* bitcast (i32* ()* @bar to float* ()*)( )
                        to label %invoke_cont unwind label %X           ; <float*> [#uses=1]

invoke_cont:            ; preds = %0
        ret float* %tmp.11

X:              ; preds = %0
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
        ret float* null
}

declare i32 @__gxx_personality_v0(...)
