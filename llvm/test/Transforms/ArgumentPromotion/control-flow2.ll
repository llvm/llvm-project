; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

; CHECK: load i32, i32* %A
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define internal i32 @callee(i1 %C, i32* %P) {
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 17

F:              ; preds = %0
        %X = load i32, i32* %P               ; <i32> [#uses=1]
        ret i32 %X
}

define i32 @foo() {
        %A = alloca i32         ; <i32*> [#uses=2]
        store i32 17, i32* %A
        %X = call i32 @callee( i1 false, i32* %A )              ; <i32> [#uses=1]
        ret i32 %X
}

