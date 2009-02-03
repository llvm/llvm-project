; This testcase checks to see if the simplifycfg pass is converting invoke
; instructions to call instructions if the handler just rethrows the exception.

; If this test is successful, the function should be reduced to 'call; ret'

; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | \
; RUN:   not egrep {\\(invoke\\)|\\(br\\)}

declare void @bar()

define i32 @test() {
        invoke void @bar( )
                        to label %Ok unwind label %Rethrow
Ok:             ; preds = %0
        ret i32 0
Rethrow:                ; preds = %0
        unwind
}
