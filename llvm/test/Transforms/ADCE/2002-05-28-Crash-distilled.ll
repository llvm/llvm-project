; This testcase is a distilled form of: 2002-05-28-Crash.ll

; RUN: opt < %s -passes=adce
; RUN: opt < %s -passes=adce -adce-remove-loops -S | FileCheck %s

define float @test(i32 %i) {
        %F = sitofp i32 %i to float             ; <float> [#uses=1]
        %I = bitcast i32 %i to i32              ; <i32> [#uses=1]
        br label %Loop

Loop:           ; preds = %Loop, %0
        %B = icmp ne i32 %I, 0          ; <i1> [#uses=1]
; CHECK:   br label %Out
        br i1 %B, label %Out, label %Loop

Out:            ; preds = %Loop
        ret float %F
}

