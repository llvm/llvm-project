; RUN: llvm-as < %s | opt -instcombine -disable-output

declare i32* @bar()

define float* @foo() {
        %tmp.11 = invoke float* bitcast (i32* ()* @bar to float* ()*)( )
                        to label %invoke_cont unwind label %X           ; <float*> [#uses=1]

invoke_cont:            ; preds = %0
        ret float* %tmp.11

X:              ; preds = %0
        ret float* null
}

