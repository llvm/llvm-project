; LoopInfo is incorrectly calculating loop nesting!  In this case it doesn't 
; figure out that loop "Inner" should be nested inside of leep "LoopHeader", 
; and instead nests it just inside loop "Top"
;
; RUN: opt < %s -passes='print<loops>' -disable-output 2>&1 | FileCheck %s

; CHECK:  Loop at depth 3 containing: %Inner<header><latch><exiting>
define void @test() {
        br label %Top

Top:            ; preds = %Out, %0
        br label %LoopHeader

Next:           ; preds = %LoopHeader
        br i1 false, label %Inner, label %Out

Inner:          ; preds = %Inner, %Next
        br i1 false, label %Inner, label %LoopHeader

LoopHeader:             ; preds = %Inner, %Top
        br label %Next

Out:            ; preds = %Next
        br i1 false, label %Top, label %Done

Done:           ; preds = %Out
        ret void
}

