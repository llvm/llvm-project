; RUN: llvm-as < %s | opt -extract-blocks -disable-output
define i32 @foo() {
        br label %EB

EB:             ; preds = %0
        %V = invoke i32 @foo( )
                        to label %Cont unwind label %Unw                ; <i32> [#uses=1]

Cont:           ; preds = %EB
        ret i32 %V

Unw:            ; preds = %EB
        unwind
}

