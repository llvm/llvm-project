; RUN: llvm-as < %s | llc -enable-correct-eh-support

define i32 @test() {
        unwind
}

define i32 @main() {
        %X = invoke i32 @test( )
                        to label %cont unwind label %EH         ; <i32> [#uses=0]

cont:           ; preds = %0
        ret i32 1

EH:             ; preds = %0
        ret i32 0
}

