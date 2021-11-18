; Test that PHI nodes and select instructions do not necessarily make stuff
; non-constant.

; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: global

@X = internal global i32 4              ; <i32*> [#uses=2]
@Y = internal global i32 5              ; <i32*> [#uses=2]

define i32 @test1(i1 %C) {
        %P = select i1 %C, i32* @X, i32* @Y             ; <i32*> [#uses=1]
        %V = load i32, i32* %P               ; <i32> [#uses=1]
        ret i32 %V
}

define i32 @test2(i1 %C) {
; <label>:0
        br i1 %C, label %T, label %Cont

T:              ; preds = %0
        br label %Cont

Cont:           ; preds = %T, %0
        %P = phi i32* [ @X, %0 ], [ @Y, %T ]            ; <i32*> [#uses=1]
        %V = load i32, i32* %P               ; <i32> [#uses=1]
        ret i32 %V
}
