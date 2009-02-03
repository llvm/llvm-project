; RUN: llvm-as < %s | llc -march=c

declare i32 @callee(i32, i32)

define i32 @test(i32 %X) {
; <label>:0
        %A = invoke i32 @callee( i32 %X, i32 5 )
                        to label %Ok unwind label %Threw                ; <i32> [#uses=1]

Ok:             ; preds = %Threw, %0
        %B = phi i32 [ %A, %0 ], [ -1, %Threw ]         ; <i32> [#uses=1]
        ret i32 %B

Threw:          ; preds = %0
        br label %Ok
}

