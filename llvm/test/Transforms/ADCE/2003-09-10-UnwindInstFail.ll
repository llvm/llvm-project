; RUN: llvm-as < %s | opt -adce -disable-output

define void @test() {
        br i1 false, label %then, label %endif

then:           ; preds = %0
        invoke void null( i8* null )
                        to label %invoke_cont unwind label %invoke_catch

invoke_catch:           ; preds = %then
        unwind

invoke_cont:            ; preds = %then
        ret void

endif:          ; preds = %0
        ret void
}

