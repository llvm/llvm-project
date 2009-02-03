; This is a basic sanity check for constant propogation.  The add instruction 
; should be eliminated.

; RUN: llvm-as < %s | opt -constprop -die | llvm-dis | not grep add

define i32 @test(i1 %B) {
        br i1 %B, label %BB1, label %BB2

BB1:            ; preds = %0
        %Val = add i32 0, 0             ; <i32> [#uses=1]
        br label %BB3

BB2:            ; preds = %0
        br label %BB3

BB3:            ; preds = %BB2, %BB1
        %Ret = phi i32 [ %Val, %BB1 ], [ 1, %BB2 ]              ; <i32> [#uses=1]
        ret i32 %Ret
}

