; Test that LICM uses basicaa to do alias analysis, which is capable of
; disambiguating some obvious cases.  If LICM is able to disambiguate the
; two pointers, then the load should be hoisted, and the store sunk.

; RUN: opt < %s -aa-pipeline=basic-aa -passes='loop-mssa(licm)' -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@A = global i32 7               ; <i32*> [#uses=3]
@B = global i32 8               ; <i32*> [#uses=2]
@C = global [2 x i32] [ i32 4, i32 8 ]          ; <[2 x i32]*> [#uses=2]

define i32 @test1(i1 %c) {
        %Atmp = load i32, i32* @A            ; <i32> [#uses=2]
        br label %Loop

Loop:           ; preds = %Loop, %0
        %ToRemove = load i32, i32* @A                ; <i32> [#uses=1]
        store i32 %Atmp, i32* @B
        br i1 %c, label %Out, label %Loop

Out:            ; preds = %Loop
        %X = sub i32 %ToRemove, %Atmp           ; <i32> [#uses=1]
        ret i32 %X
        
; The Loop block should be empty after the load/store are promoted.
; CHECK:     @test1
; CHECK:        load i32, i32* @A
; CHECK:         load i32, i32* @A
; CHECK:         store i32 %Atmp, i32* @B
; CHECK:      Loop:
; CHECK-NEXT:   br i1 %c, label %Out, label %Loop
; CHECK:      Out:
}

define i32 @test2(i1 %c) {
        br label %Loop

Loop:           ; preds = %Loop, %0
        %AVal = load i32, i32* @A            ; <i32> [#uses=2]
        %C0 = getelementptr [2 x i32], [2 x i32]* @C, i64 0, i64 0         ; <i32*> [#uses=1]
        store i32 %AVal, i32* %C0
        %BVal = load i32, i32* @B            ; <i32> [#uses=2]
        %C1 = getelementptr [2 x i32], [2 x i32]* @C, i64 0, i64 1         ; <i32*> [#uses=1]
        store i32 %BVal, i32* %C1
        br i1 %c, label %Out, label %Loop

Out:            ; preds = %Loop
        %X = sub i32 %AVal, %BVal               ; <i32> [#uses=1]
        ret i32 %X
; The Loop block should be empty after the load/store are promoted.
; CHECK:     @test2
; CHECK:      Loop:
; CHECK-NEXT:   br i1 %c, label %Out, label %Loop
}

