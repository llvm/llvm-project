; RUN: opt -S -indvars %s | FileCheck %s

; The indvar simplification code should ensure that the first PHI in the block 
; is the canonical one!

define i32 @test() {
; <label>:0
        br label %Loop

Loop:           ; preds = %Loop, %0
; CHECK: Loop:
; CHECK-NEXT: Canonical
        %NonIndvar = phi i32 [ 200, %0 ], [ %NonIndvarNext, %Loop ]             ; <i32> [#uses=1]
        %Canonical = phi i32 [ 0, %0 ], [ %CanonicalNext, %Loop ]               ; <i32> [#uses=2]
        store i32 %Canonical, i32* null
        %NonIndvarNext = sdiv i32 %NonIndvar, 2         ; <i32> [#uses=1]
        %CanonicalNext = add i32 %Canonical, 1          ; <i32> [#uses=1]
        br label %Loop
}

